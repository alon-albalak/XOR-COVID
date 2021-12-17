# largest language subsets:
# es, zh, fr, ja, ar, de, ru, ko, vi, pt, it

# multiple settings
# 1. Translate all non-english query into english
# 2. Translate english answers into other languages

# An interesting test for this set up would be to select negative paragraphs only from the query language, or at least mostly from the query language

import jsonlines
import random
import argparse

from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer
from torch.cuda import is_available
cuda = "cuda" if is_available() else "cpu"

def translate_COUGH(data_path, save_path, batch_size=1, neg_para_samples=30):
    # format datums with keys: ['question', 'answers', 'id', 'pos_paras', 'neg_paras', 'language']
    langs = ["es","zh","fr","ja","ar","de","ru","ko","vi","pt","it"]


    with jsonlines.open(data_path) as reader:
        original_data = [datum for datum in reader]

    translated_data = []
    original_data_sorted = {lang: [datum for datum in original_data if datum['language'] == lang] for lang in langs}
    english_data = [datum for datum in original_data if datum['language'] == "en"]

    for lang in langs:
        # translate english answers into other languages
        # select negative paragraphs from original data
        print(f"Translating answers to {lang}")
        translation_model = f"Helsinki-NLP/opus-mt-en-{lang}"
        try:
            tokenizer = MarianTokenizer.from_pretrained(translation_model)
        except OSError:
            print(f"No english to {lang} model found")
            continue
        model = MarianMTModel.from_pretrained(translation_model)
        model.to(cuda)
        model.eval()

        for i in tqdm(range(0, len(english_data), batch_size)):
            # gather query/answers
            questions = [datum['question'] for datum in english_data[i:i+batch_size]]
            answers = [datum['answers'][0] for datum in english_data[i:i+batch_size]]
            titles = [datum['pos_paras'][0]['title'] for datum in english_data[i:i+batch_size]]
            ids = [datum['id'] for datum in english_data[i:i + batch_size]]
            translated_answers = model.generate(**tokenizer(answers, truncation=True, padding=True, return_tensors="pt").to(cuda))
            translated_answers = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in translated_answers]

            for q,a,t,original_answer,id in zip(questions, translated_answers, titles, answers,ids):
                sample = {
                    "question":q,
                    "answers":[a],
                    "id":id,
                    "pos_paras":[{"title":t,"text":a}],
                    "neg_paras":[],
                    "language":f"en|{lang}"
                }
                while len(sample['neg_paras']) < neg_para_samples:
                    random_choice = random.choice(original_data)
                    if random_choice['answers'][0] != original_answer:
                        sample['neg_paras'].append(
                            {
                                "title":random_choice['pos_paras'][0]['title'],
                                "text":random_choice['pos_paras'][0]['text'],
                                "id":random_choice['id']
                            }
                        )
                translated_data.append(sample)

    # translate non-english queries into english
    # also select negative paragraphs from original data
    for lang, data in original_data_sorted.items():
        print(f"Translating questions from {lang} to en")
        translation_model = f"Helsinki-NLP/opus-mt-{lang}-en"
        try:
            tokenizer = MarianTokenizer.from_pretrained(translation_model)
        except OSError:
            print(f"No {lang} to en model found")
            continue
        model = MarianMTModel.from_pretrained(translation_model)
        model.to(cuda)
        model.eval()
        for i in tqdm(range(0, len(data), batch_size)):
            # gather query/answers
            questions = [datum['question'] for datum in data[i:i+batch_size]]
            answers = [datum['answers'][0] for datum in data[i:i+batch_size]]
            titles = [datum['pos_paras'][0]['title'] for datum in data[i:i+batch_size]]
            ids = [datum['id'] for datum in english_data[i:i + batch_size]]
            translated_questions = model.generate(**tokenizer(questions, truncation=True, padding=True, return_tensors="pt").to(cuda))
            translated_questions = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in translated_questions]

            for q,a,t,original_question,id in zip(translated_questions, answers, titles, questions,ids):
                sample = {
                    "question":q,
                    "answers":[a],
                    "id":id,
                    "pos_paras":[{"title":t,"text":a}],
                    "neg_paras":[],
                    "language":f"{lang}|en"
                }
                while len(sample['neg_paras']) < neg_para_samples:
                    random_choice = random.choice(original_data)
                    if random_choice["question"] != original_question:
                        sample['neg_paras'].append(
                            {
                                "title":random_choice['pos_paras'][0]['title'],
                                "text":random_choice['pos_paras'][0]['text'],
                                "id":random_choice['id']
                            }
                        )
                translated_data.append(sample)
    translated_data.extend(original_data)
    
    with jsonlines.open(save_path, "w") as writer:
        writer.write_all(translated_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--batch_size",type=int, default=50)


    args = parser.parse_args()

    translate_COUGH(args.data_path, args.save_path, batch_size=args.batch_size)