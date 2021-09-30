import os
import json
import re
import random
from tqdm import tqdm
import numpy as np
from transformers import MarianMTModel, MarianTokenizer
from torch.cuda import is_available

cuda = "cuda" if is_available() else "cpu"
langs_without_spaces = ["th", "zh"]

def levenshtein_ratio(s, t):
    """ levenshtein_ratio_and_distance:
        Calculates levenshtein distance between two strings.
        If ratio_calc = True, the function computes the
        levenshtein distance ratio of similarity between two strings
        For all i and j, distance[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
    """
    # Initialize matrix of zeros
    rows = len(s)+1
    cols = len(t)+1
    distance = np.zeros((rows,cols),dtype = int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1,cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions    
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                cost = 2
            distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
                                 distance[row][col-1] + 1,          # Cost of insertions
                                 distance[row-1][col-1] + cost)     # Cost of substitutions
    if (not s) or (not t):
        return 0
    # Computation of the Levenshtein Distance Ratio
    Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
    return Ratio

def get_answer_by_levenshtein_ratio(context, answer, lang,skip_length=3):
    split_on_space = lang not in langs_without_spaces
    if split_on_space:
        split_context = context.split(" ")
        answer_length = len(answer.split(" "))
    else:
        split_context = context
        answer_length = len(answer)
    
    answer_index=-1
    ratio_best = 0
    skip_next=False

    for i in range(0, len(split_context), skip_length):
        if skip_next:
            skip_next=False
            continue

        if split_on_space:
            subcontext = " ".join(split_context[i:i+answer_length])
        else:
            subcontext = split_context[i:i+answer_length]

        ratio = levenshtein_ratio(answer, subcontext)
        if ratio > ratio_best:
            ratio_best = ratio
            answer_index = i

        if ratio >= ratio_best*0.9:
            for j in range(1,skip_length):
                if split_on_space:
                    subcontext = " ".join(split_context[i+j:i+j+answer_length])
                else:
                    subcontext = split_context[i+j:i+j+answer_length]
                ratio = levenshtein_ratio(answer, subcontext)
                if ratio > ratio_best:
                    ratio_best = ratio
                    answer_index = i+j
        else:
            skip_next=True
    if split_on_space:
        answer = " ".join(split_context[answer_index:answer_index+answer_length])
        answer_start = context.find(answer)
        if answer_start == -1:
            print(f"Could not find answer")
    else:
        answer = split_context[answer_index:answer_index+answer_length]
        answer_start = answer_index

    return answer, answer_start, ratio_best


def translate_covidQA_w_anchors(data_path="data",save_path="multilingual_covidQA",langs=["es"],batch_size=1):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    a1 = " *1* "
    a2 = " *2* "
    data = []

    for split in ["train","dev","test"]:
        d = json.load(open(os.path.join(data_path,f"qa_{split}.json")))
        data.extend(d['data'])

    for datum in data:
        answer = datum['answers'][0]['text']
        start = datum['answers'][0]['answer_start']
        context = datum['context']
        end = start+len(answer)
        datum['anchored_context'] = context[:start]+a1+answer+a2+context[end:]

    for lang in langs:
        print(f"Translating to {lang}")
        translated_data = {"data":[]}
        missed_anchors = 0
        ratios = []
        translation_model = f"Helsinki-NLP/opus-mt-en-{lang}"
        try:
            tokenizer = MarianTokenizer.from_pretrained(translation_model)
        except OSError:
            print(f"No english to {lang} model found")
            continue
        model = MarianMTModel.from_pretrained(translation_model)
        model.to(cuda)
        model.eval()

        answers_not_found = 0
        for i in tqdm(range(0,len(data), batch_size)):
            # translate full contexts with anchors
            anchored_contexts = [datum['anchored_context'] for datum in data[i:i+batch_size]]
            translated_anchored_context_tokens = model.generate(**tokenizer(anchored_contexts, truncation=True, padding=True, return_tensors="pt").to(cuda))
            translated_anchored_contexts = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in translated_anchored_context_tokens]
            # translate questions
            questions = [datum['question'] for datum in data[i:i+batch_size]]
            translated_question_tokens = model.generate(**tokenizer(questions, truncation=True, padding=True, return_tensors="pt").to(cuda))
            translated_questions = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in translated_question_tokens]
            # translate answers
            answers = [datum['answers'][0]['text'] for datum in data[i:i+batch_size]]
            
            translated_answers = []
            translated_answer_starts = []
            translated_contexts = []
            
            for j, tc in enumerate(translated_anchored_contexts):
                extracted_translated_answer = ""
                answer_start = tc.find(a1.strip())
                answer_end = tc.find(a2.strip())
                # if the answer is found
                if answer_start >-1 and answer_end>-1:
                    extracted_translated_answer = tc[answer_start+len(a1.strip()):answer_end].strip()
                    # if extracted_translated_answer not in translated_anchored_contexts[j]:
                    #     answers_not_found += 1
                translated_answers.append(extracted_translated_answer)
                # try to find the answer after removing the anchor points
                translated_context = translated_anchored_contexts[j].replace("*1*","").replace("*2*","")
                translated_context = re.sub(" {2,}", " ", translated_context)
                if extracted_translated_answer:
                    answer_start = translated_context.find(extracted_translated_answer)
                else:
                    answer_start = -1
                translated_answer_starts.append(answer_start)
                translated_contexts.append(translated_context)
            
            for j in range(len(anchored_contexts)):
                # if answer was not found, use levelshtein distance
                if any([anchor.strip() not in translated_anchored_contexts[j] for anchor in [a1,a2]]) or translated_answer_starts[j] == -1:
                    missed_anchors += 1
                    translated_context_and_answer_tokens = model.generate(**tokenizer([data[i+j]['context'],answers[j]], truncation=True, padding=True, return_tensors="pt").to(cuda))
                    translated_context = tokenizer.decode(translated_context_and_answer_tokens[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    translated_answer = tokenizer.decode(translated_context_and_answer_tokens[1], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    translated_contexts[j] = translated_context
                    answer, answer_start, ratio = get_answer_by_levenshtein_ratio(translated_context,translated_answer, lang=lang)
                    ratios.append(ratio)
                    if ratio < 0.25:
                        answers_not_found += 1
                        continue
                    translated_answers[j] = answer
                    translated_answer_starts[j] = answer_start

                if not translated_answers[j] or (translated_answers[j] not in translated_contexts[j]) or translated_answer_starts[j] == -1:
                    answers_not_found += 1
                    continue
                translated_datum = {
                    "title": data[i+j]['title'],
                    "original_context":data[i+j]['context'],
                    "original_question":data[i+j]['question'],
                    "original_answers":data[i+j]['answers'],
                    "context":translated_contexts[j],
                    "question": translated_questions[j],
                    "answers":[{
                        "text":translated_answers[j],
                        "answer_start":translated_answer_starts[j]
                    }]
                }
                translated_data['data'].append(translated_datum)

        print(f"FINISHED {lang}")
        print(f"MEAN RATIO: {np.mean(ratios)}")
        print(f"MIN RATIO: {np.min(ratios)}")
        print(f"std ratios: {np.std(ratios)}")
        with open(os.path.join(save_path,f"qa_{lang}.json"), "w", encoding="utf8") as f:
            json.dump(translated_data, f, indent=2, ensure_ascii=False)
        print(f"MISSED ANCHORS: {missed_anchors}")
        print(f"COULD NOT FIND {answers_not_found} answers")

def combine_split_covidQA_data(data_path="multilingual_covidQA",
        one2one_path="multilingual_covidQA",en2many_path="e2m_covidQA",one2one_en2many_path="one2one_e2m",many2many_path="m2m_covidQA",
        one2one=True,en2many=True,one2one_en2many=True,many2many=True,dev_percent=15,test_percent=15, seed=10):
    """
    will include datums with all possible language pairs for question_{L1} : answer_{L2}
    """
    random.seed(seed)

    full_dataset = []

    for lang in langs:
        lang_path=os.path.join(data_path, f"qa_{lang}.json")
        if not os.path.exists(lang_path):
            continue
        covidqa_data = json.load(open(lang_path))
        for datum in covidqa_data['data']:
            datum['lang'] = lang
        full_dataset.extend(covidqa_data['data'])

    data_by_question = {}
    for datum in full_dataset:
        if datum['original_question'] not in data_by_question:
            # automatically add the english question, context, and answers
            data_by_question[datum['original_question']] = {
                "title":datum['title'],
                "questions":{"en":datum['original_question']},
                "context_answers":{"en":[datum['original_context'],datum['original_answers']]},
            }
        data_by_question[datum['original_question']]['questions'][datum['lang']] = datum['question']
        data_by_question[datum['original_question']]['context_answers'][datum['lang']] = [datum['context'],datum['answers']]

    questions = list(data_by_question.keys())
    random.shuffle(questions)

    num_test_questions = len(questions)*test_percent//100
    num_dev_questions = len(questions)*dev_percent//100
    test_subset_questions = questions[:num_test_questions]
    dev_subset_questions = questions[num_test_questions:(num_test_questions+num_dev_questions)]
    train_subset_questions = questions[(num_test_questions+num_dev_questions):]

    # create one-to-one data
    if one2one:
        if not os.path.isdir(one2one_path):
            os.mkdir(one2one_path)
        one2one_train = {'data': []}
        one2one_dev = {'data': []}
        one2one_test = {'data': []}
        one2one_id = 0
        for question in train_subset_questions:
            for lang in data_by_question[question]['questions']:
                datum = {
                    "id": str(one2one_id),
                    "title": data_by_question[question]['title'],
                    "context":data_by_question[question]['context_answers'][lang][0],
                    "question":data_by_question[question]['questions'][lang],
                    "answers":data_by_question[question]['context_answers'][lang][1],
                    "lang":lang
                }
                one2one_id += 1
                one2one_train['data'].append(datum)
        for question in dev_subset_questions:
            for lang in data_by_question[question]['questions']:
                datum = {
                    "id": str(one2one_id),
                    "title": data_by_question[question]['title'],
                    "context":data_by_question[question]['context_answers'][lang][0],
                    "question":data_by_question[question]['questions'][lang],
                    "answers":data_by_question[question]['context_answers'][lang][1],
                    "lang":lang
                }
                one2one_id += 1
                one2one_dev['data'].append(datum)
        for question in test_subset_questions:
            for lang in data_by_question[question]['questions']:
                datum = {
                    "id": str(one2one_id),
                    "title": data_by_question[question]['title'],
                    "context":data_by_question[question]['context_answers'][lang][0],
                    "question":data_by_question[question]['questions'][lang],
                    "answers":data_by_question[question]['context_answers'][lang][1],
                    "lang":lang
                }
                one2one_id += 1
                one2one_test['data'].append(datum)

        with open(os.path.join(one2one_path,"qa_train.json"), "w",encoding='utf8') as f:
            json.dump(one2one_train, f, indent=2, ensure_ascii=False)

        with open(os.path.join(one2one_path,"qa_dev.json"), "w",encoding='utf8') as f:
            json.dump(one2one_dev, f, indent=2, ensure_ascii=False)

        with open(os.path.join(one2one_path,"qa_test.json"), "w",encoding='utf8') as f:
            json.dump(one2one_test, f, indent=2, ensure_ascii=False)

    # create english-to-many data
    if en2many:
        if not os.path.isdir(en2many_path):
            os.mkdir(en2many_path)
        en2many_train = {'data': []}
        en2many_dev = {'data': []}
        en2many_test = {'data': []}
        en2many_id = 0
        for question in train_subset_questions:
            for lang in data_by_question[question]['questions']:
                datum = {
                    "id": str(en2many_id),
                    "title": data_by_question[question]['title'],
                    "context":data_by_question[question]['context_answers'][lang][0],
                    "question":data_by_question[question]['questions']["en"],
                    "answers":data_by_question[question]['context_answers'][lang][1],
                    "q_lang":"en",
                    "c_lang":lang
                }
                en2many_id += 1
                en2many_train['data'].append(datum)

        for question in dev_subset_questions:
            for lang in data_by_question[question]['questions']:
                datum = {
                    "id": str(en2many_id),
                    "title": data_by_question[question]['title'],
                    "context":data_by_question[question]['context_answers'][lang][0],
                    "question":data_by_question[question]['questions']["en"],
                    "answers":data_by_question[question]['context_answers'][lang][1],
                    "q_lang":"en",
                    "c_lang":lang
                }
                en2many_id += 1
                en2many_dev['data'].append(datum)

        for question in test_subset_questions:
            for lang in data_by_question[question]['questions']:
                datum = {
                    "id": str(en2many_id),
                    "title": data_by_question[question]['title'],
                    "context":data_by_question[question]['context_answers'][lang][0],
                    "question":data_by_question[question]['questions']["en"],
                    "answers":data_by_question[question]['context_answers'][lang][1],
                    "q_lang":"en",
                    "c_lang":lang
                }
                en2many_id += 1
                en2many_test['data'].append(datum)

        with open(os.path.join(en2many_path,"qa_train.json"), "w",encoding='utf8') as f:
            json.dump(en2many_train, f, indent=2, ensure_ascii=False)

        with open(os.path.join(en2many_path,"qa_dev.json"), "w",encoding='utf8') as f:
            json.dump(en2many_dev, f, indent=2, ensure_ascii=False)

        with open(os.path.join(en2many_path,"qa_test.json"), "w",encoding='utf8') as f:
            json.dump(en2many_test, f, indent=2, ensure_ascii=False)   

    # create one-to-to and english-to-many data
    if one2one_en2many:
        if not os.path.isdir(one2one_en2many_path):
            os.mkdir(one2one_en2many_path)
        one2one_en2many_train = {'data': []}
        one2one_en2many_dev = {'data': []}
        one2one_en2many_test = {'data': []}
        one2one_en2many_id = 0
        for question in train_subset_questions:
            for lang in data_by_question[question]['questions']:
                datum = {
                    "id": str(one2one_en2many_id),
                    "title": data_by_question[question]['title'],
                    "context":data_by_question[question]['context_answers'][lang][0],
                    "question":data_by_question[question]['questions'][lang],
                    "answers":data_by_question[question]['context_answers'][lang][1],
                    "q_lang":lang,
                    "c_lang":lang
                }
                one2one_en2many_id += 1
                one2one_en2many_train['data'].append(datum)

                if lang != "en":
                    datum = {
                        "id": str(one2one_en2many_id),
                        "title": data_by_question[question]['title'],
                        "context":data_by_question[question]['context_answers'][lang][0],
                        "question":data_by_question[question]['questions']["en"],
                        "answers":data_by_question[question]['context_answers'][lang][1],
                        "q_lang":"en",
                        "c_lang":lang
                    }
                    one2one_en2many_id += 1
                    one2one_en2many_train['data'].append(datum)
        for question in dev_subset_questions:
            for lang in data_by_question[question]['questions']:
                datum = {
                    "id": str(one2one_en2many_id),
                    "title": data_by_question[question]['title'],
                    "context":data_by_question[question]['context_answers'][lang][0],
                    "question":data_by_question[question]['questions'][lang],
                    "answers":data_by_question[question]['context_answers'][lang][1],
                    "q_lang":lang,
                    "c_lang":lang
                }
                one2one_en2many_id += 1
                one2one_en2many_dev['data'].append(datum)

                if lang != "en":
                    datum = {
                        "id": str(one2one_en2many_id),
                        "title": data_by_question[question]['title'],
                        "context":data_by_question[question]['context_answers'][lang][0],
                        "question":data_by_question[question]['questions']["en"],
                        "answers":data_by_question[question]['context_answers'][lang][1],
                        "q_lang":"en",
                        "c_lang":lang
                    }
                    one2one_en2many_id += 1
                    one2one_en2many_dev['data'].append(datum)
        for question in test_subset_questions:
            for lang in data_by_question[question]['questions']:
                datum = {
                    "id": str(one2one_en2many_id),
                    "title": data_by_question[question]['title'],
                    "context":data_by_question[question]['context_answers'][lang][0],
                    "question":data_by_question[question]['questions'][lang],
                    "answers":data_by_question[question]['context_answers'][lang][1],
                    "q_lang":lang,
                    "c_lang":lang
                }
                one2one_en2many_id += 1
                one2one_en2many_test['data'].append(datum)

                if lang != "en":
                    datum = {
                        "id": str(one2one_en2many_id),
                        "title": data_by_question[question]['title'],
                        "context":data_by_question[question]['context_answers'][lang][0],
                        "question":data_by_question[question]['questions']["en"],
                        "answers":data_by_question[question]['context_answers'][lang][1],
                        "q_lang":"en",
                        "c_lang":lang
                    }
                    one2one_en2many_id += 1
                    one2one_en2many_test['data'].append(datum)

        with open(os.path.join(one2one_en2many_path,"qa_train.json"), "w",encoding='utf8') as f:
            json.dump(one2one_en2many_train, f, indent=2, ensure_ascii=False)

        with open(os.path.join(one2one_en2many_path,"qa_dev.json"), "w",encoding='utf8') as f:
            json.dump(one2one_en2many_dev, f, indent=2, ensure_ascii=False)

        with open(os.path.join(one2one_en2many_path,"qa_test.json"), "w",encoding='utf8') as f:
            json.dump(one2one_en2many_test, f, indent=2, ensure_ascii=False)       

    # create many-to-many data
    if many2many:
        if not os.path.isdir(many2many_path):
            os.mkdir(many2many_path)
        many2many_train = {'data': []}
        many2many_dev = {'data': []}
        many2many_test = {'data': []}
        many2many_id = 0
        for question in train_subset_questions:
            for q_lang in data_by_question[question]['questions']:
                for c_lang in data_by_question[question]['context_answers']:
                    datum = {
                        "id":str(many2many_id),
                        "title":data_by_question[question]['title'],
                        "context":data_by_question[question]['context_answers'][c_lang][0],
                        "question":data_by_question[question]['questions'][q_lang],
                        "answers":data_by_question[question]['context_answers'][c_lang][1],
                        "q_lang":q_lang,
                        "c_lang":c_lang
                    }
                    many2many_id += 1
                    many2many_train['data'].append(datum)

        for question in dev_subset_questions:
            for q_lang in data_by_question[question]['questions']:
                for c_lang in data_by_question[question]['context_answers']:
                    datum = {
                        "id":str(many2many_id),
                        "title":data_by_question[question]['title'],
                        "context":data_by_question[question]['context_answers'][c_lang][0],
                        "question":data_by_question[question]['questions'][q_lang],
                        "answers":data_by_question[question]['context_answers'][c_lang][1],
                        "q_lang":q_lang,
                        "c_lang":c_lang
                    }
                    many2many_id += 1
                    many2many_dev['data'].append(datum)

        for question in test_subset_questions:
            for q_lang in data_by_question[question]['questions']:
                for c_lang in data_by_question[question]['context_answers']:
                    datum = {
                        "id":str(many2many_id),
                        "title":data_by_question[question]['title'],
                        "context":data_by_question[question]['context_answers'][c_lang][0],
                        "question":data_by_question[question]['questions'][q_lang],
                        "answers":data_by_question[question]['context_answers'][c_lang][1],
                        "q_lang":q_lang,
                        "c_lang":c_lang
                    }
                    many2many_id += 1
                    many2many_test['data'].append(datum)

        with open(os.path.join(many2many_path,"qa_train.json"), "w",encoding='utf8') as f:
            json.dump(many2many_train, f, indent=2, ensure_ascii=False)

        with open(os.path.join(many2many_path,"qa_dev.json"), "w",encoding='utf8') as f:
            json.dump(many2many_dev, f, indent=2, ensure_ascii=False)

        with open(os.path.join(many2many_path,"qa_test.json"), "w",encoding='utf8') as f:
            json.dump(many2many_test, f, indent=2, ensure_ascii=False)

langs = ['ar','de','el','es','hi','ro','ru','th','tr','vi','zh']

batch_size=50

if __name__ == "__main__":
    translate_covidQA_w_anchors(data_path="data", save_path="multilingual_covidQA",langs=langs,batch_size=batch_size)
    combine_split_covidQA_data(data_path="multilingual_covidQA",
            one2one_path="multilingual_covidQA/one2one",
            en2many_path="multilingual_covidQA/en2many",
            one2one_en2many_path="multilingual_covidQA/one2one_en2many",
            many2many_path="multilingual_covidQA/many2many")
