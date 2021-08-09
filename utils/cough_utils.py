import sys
import os
import csv
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import jsonlines
from tqdm import tqdm

from transformers import MarianMTModel, MarianTokenizer

csv.field_size_limit(sys.maxsize)


def load_csv_to_dicts(csv_path):
    """Load a csv into dict"""
    print(f"Loading data from {csv_path}")
    with open(csv_path, "r") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                data = {r: [] for r in row}
                print(f"column names are {', '.join(row)}")
            else:
                for r in data:
                    data[r].append(row[r])
            line_count += 1
    print(f"processed {line_count} lines")
    return data


def load_csv_to_list(csv_path):
    fields, rows = [], []
    with open(csv_path, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        fields = next(csvreader)
        for row in csvreader:
            rows.append(row)
    return fields, rows


def dataset_statistics():
    qb = load_csv_to_dicts("../COUGH/User_Query_Bank.csv")
    fb = load_csv_to_dicts("../COUGH/FAQ_Bank.csv")

    q_lens = [len(q.split()) for q in qb["query"]]
    a_lens = [len(a.split()) for a in fb["answer"]]
    print(f"Mean query length: {sum(q_lens)/len(q_lens)}")
    print(f"Max query length: {max(q_lens)}")
    print(f"Mean answer length: {sum(a_lens)/len(a_lens)}")
    print(f"Max answer length: {max(a_lens)}")

    print(f"Num queries over 50 words: {sum([1 if q > 50 else 0 for q in q_lens ])}")
    print(f"Num answers under 150 words: {sum([1 if a < 150 else 0 for a in a_lens ])}")
    print(f"Num answers over 250 words: {sum([1 if a > 250 else 0 for a in a_lens ])}")

    q_ar = np.array(q_lens)
    a_ar = np.array(a_lens)
    plt.hist(q_ar, bins=[0, 5, 10, 15, 20, 25, 30, 40, 50])
    plt.title("Question lengths")
    plt.show()

    plt.hist(a_ar, bins=[0, 50, 100, 150, 200, 300, 400, 500])
    plt.title("Answer lengths")
    plt.show()

    COUGH_fields, COUGH_data = load_csv_to_list("COUGH/FAQ_Bank.csv")
    lang_idx = COUGH_fields.index("language")
    langs = {}
    for datum in COUGH_data:
        if datum[lang_idx] not in langs:
            langs[datum[lang_idx]] = 0
        langs[datum[lang_idx]] += 1
    langs = {k: v for k, v in sorted(langs.items(), key=lambda item: item[1], reverse=True)}
    print(f"LANGS: {langs}")


def convert_COUGH_to_retrieval_dataset(
    data_path="../COUGH/FAQ_Bank.csv", dev_percent=15, test_percent=15, neg_para_samples=30
):
    """
    converts the COUGH dataset into a .txt file with each line as a json object
    each line contains:
    {
        "question": str,
        "answers": list of str,
        "id":str articleID,
        "pos_paras": list of dict [{"title":str,"text":str},...],
        "neg_paras": list of dict [{"title":str,"text":str, "id":str},...],
    }
    """
    COUGH_fields, COUGH_data = load_csv_to_list(data_path)
    lang_idx = COUGH_fields.index("language")
    train_subset, dev_subset, test_subset = [], [], []

    # group data by language
    langs = {}
    for datum in COUGH_data:
        if datum[lang_idx] not in langs:
            langs[datum[lang_idx]] = []
        langs[datum[lang_idx]].append(datum)

    # split data into train/dev/test, preference is to make sure we get diverse language samples in test, then dev, then train sets
    for lang in langs:
        random.shuffle(langs[lang])
        num_test_samples = len(langs[lang]) * test_percent // 100
        num_dev_samples = len(langs[lang]) * dev_percent // 100
        test_subset.extend(langs[lang][:num_test_samples])
        dev_subset.extend(langs[lang][num_test_samples : (num_test_samples + num_dev_samples)])
        train_subset.extend(langs[lang][(num_dev_samples + num_test_samples) :])

    train_data, dev_data, test_data = [], [], []

    # gather and organize data
    for subset, data in [[train_subset, train_data], [dev_subset, dev_data], [test_subset, test_data]]:

        for datum in subset:
            sample = {
                "question": datum[COUGH_fields.index("question")],
                "answers": [datum[COUGH_fields.index("answer")]],
                "id": datum[COUGH_fields.index("index")],
                "pos_paras": [{"title": datum[COUGH_fields.index("url")], "text": datum[COUGH_fields.index("answer")]}],
                "neg_paras": [],
            }
            while len(sample["neg_paras"]) < neg_para_samples:
                random_choice = random.choice(train_subset)
                if random_choice[COUGH_fields.index("index")] != datum[COUGH_fields.index("index")]:
                    sample["neg_paras"].append(
                        {
                            "title": random_choice[COUGH_fields.index("url")],
                            "text": random_choice[COUGH_fields.index("answer")],
                            "id": random_choice[COUGH_fields.index("index")],
                        }
                    )
            data.append(sample)

    return train_data, dev_data, test_data


def create_save_COUGH_retrieval_data(
    save_path="../COUGH/", data_path="../COUGH/FAQ_Bank.csv", dev_percent=15, test_percent=15, neg_para_samples=30
):
    train_data, dev_data, test_data = convert_COUGH_to_retrieval_dataset(
        data_path, dev_percent, test_percent, neg_para_samples
    )
    train_path = os.path.join(save_path, "retrieval_train.txt")
    dev_path = os.path.join(save_path, "retrieval_dev.txt")
    test_path = os.path.join(save_path, "retrieval_test.txt")

    with open(train_path, "w", encoding="utf8") as file:
        for datum in train_data:
            json.dump(datum, file, ensure_ascii=False)
            file.write("\n")

    with open(dev_path, "w", encoding="utf8") as file:
        for datum in dev_data:
            json.dump(datum, file, ensure_ascii=False)
            file.write("\n")

    with open(test_path, "w", encoding="utf8") as file:
        for datum in test_data:
            json.dump(datum, file, ensure_ascii=False)
            file.write("\n")

def read_jsonlines(file_name):
    lines = []
    print("loading examples from {0}".format(file_name))
    with jsonlines.open(file_name) as reader:
        for obj in reader:
            lines.append(obj)
    return lines

def translate_COUGH_questions(save_path="../COUGH/retrieval_test_all_langs.txt", data_path="../COUGH/retrieval_test.txt"):

    # get full list of language pairs
    translation_langs = ["ar","de","en","es","fr","ru","zh"]
    translation_pairs = [[source_lang,[target_lang for target_lang in translation_langs if target_lang != source_lang]] for source_lang in translation_langs]

    translated_data = []
    data = read_jsonlines(data_path)


    for item in tqdm(data):
        lang = item['language']
        new_item = {
            f'question_{lang}': item['question'],
            'answers': item['answers'],
            'id': item['id'],
            'pos_paras': item['pos_paras'],
            'neg_paras': item['neg_paras'],
            "original_lang":item['language']
        }
        translated_data.append(new_item)
        

    model_path = "Helsinki-NLP/opus-mt-{}-{}"
    for translation_pair in translation_pairs:
        source_lang = translation_pair[0]
        for target_lang in translation_pair[1]:
            try:
                mt_tokenizer = MarianTokenizer.from_pretrained(model_path.format(source_lang,target_lang))
            except OSError:
                print(f"No {source_lang} to {target_lang} model found")
                continue
            mt_model = MarianMTModel.from_pretrained(model_path.format(source_lang,target_lang))
            mt_model.to('cuda')
            mt_model.eval()


            for item in tqdm(translated_data, desc = f"{source_lang} to {target_lang}"):
                if item['original_lang'] == source_lang:
                    translation = mt_model.generate(**mt_tokenizer([item[f'question_{item["original_lang"]}']], truncation=True, return_tensors='pt').to('cuda'))
                    translated_q = mt_tokenizer.decode(translation[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    item[f'question_{target_lang}'] = translated_q

    with jsonlines.open(save_path, "w") as writer:
        writer.write_all(translated_data)