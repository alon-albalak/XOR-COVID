import os
import sys
import json
import random
import csv
import argparse

csv.field_size_limit(sys.maxsize)


def load_csv_to_list(csv_path):
    fields, rows = [], []
    with open(csv_path, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        fields = next(csvreader)
        for row in csvreader:
            rows.append(row)
    return fields, rows


def convert_COUGH_to_retrieval_dataset(data_path="COUGH/FAQ_Bank.csv", dev_percent=15, test_percent=15, neg_para_samples=30):
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


def create_save_COUGH_retrieval_data(save_path="COUGH/", data_path="COUGH/FAQ_Bank.csv", dev_percent=15, test_percent=15, neg_para_samples=30):
    train_data, dev_data, test_data = convert_COUGH_to_retrieval_dataset(data_path, dev_percent, test_percent, neg_para_samples)
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


parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str)
parser.add_argument("--data_path", type=str)
parser.add_argument("--dev_percent", type=int, default=15)
parser.add_argument("--test_percent", type=int, default=15)
parser.add_argument("--neg_para_samples", type=int, default=30)

args = parser.parse_args()

create_save_COUGH_retrieval_data(args.save_path, args.data_path, args.dev_percent, args.test_percent, args.neg_para_samples)
