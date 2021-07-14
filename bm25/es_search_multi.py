from elasticsearch import Elasticsearch
import argparse
import os
from doc_db import DocDB
import re
from tqdm import tqdm
import jsonlines
import json

from collections import defaultdict

import nltk
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
modelTransf = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

from utils import peraton_lang_id_to_ISO6391
LANGS=peraton_lang_id_to_ISO6391.values()

def read_jsonlines(file_name):
    lines = []
    print("loading examples from {0}".format(file_name))
    with jsonlines.open(file_name) as reader:
        for obj in reader:
            lines.append(obj)
    return lines

def search_es(es_obj, index_name, question_text, n_results=5):
    # construct query
    query = {
            'query': {
                'match': {
                    'document_text': question_text
                    }
                }
            }
    
    res = es_obj.search(index=index_name, body=query, size=n_results)
    
    return res

def embed_sentences(sentences):
    return modelTransf.encode(sentences, show_progress_bar=False)

def sentence_tokenize(text, lang):
    langs = {
        "de":"german",
        "es":"spanish",
        "fr":"french",
        "it":"italian",
        "nl":"dutch",
        "pl":"polish",
        "pt":"portuguese",
        "ru":"russian",
        "tr":"turkish",
    }
    if lang == "en":
        sentences = nltk.sent_tokenize(text)
    elif lang in langs:
        tokenizer = nltk.data.load(f"tokenizers/punkt/{langs[lang]}.pickle")
        sentences = tokenizer.tokenize(text)
    else:
        if lang == "zh":
            sentences = [sent + "\u3002" for sent in text.split("\u3002") if len(sent) > 3]
        elif lang == "ar":
            sentences = [sent + "." for sent in text.split(".") if len(sent) > 3]  # probably not great
        elif lang == "ko":
            sentences = [sent + "." for sent in text.split(".") if len(sent) > 3]  # probably not great
        else:
            sentences = text.split(".")            

    stripSentences = [stri.strip() for stri in sentences]
    return stripSentences

def cosine_similar(query_embedding, corpus_embeddings):
    topRank = min(2, corpus_embeddings.shape[0])
    cos_score = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    best_result = torch.topk(cos_score, k=topRank)
    for idx, score in zip(best_result[1], best_result[0]):
        if 0.65 <= score:
            return True
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_name_prefix', type=str, required=True,
                        help='Path to index')
    parser.add_argument('--input_data_file_name', type=str, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--output_fp', type=str)

    args = parser.parse_args()

    # args.index_name_prefix="test_index"
    # args.input_data_file_name="COUGH/retrieval_dev.txt"
    # args.port=9200

    results = {lang:[] for lang in LANGS}
    overall_results = []

    input_data = read_jsonlines(args.input_data_file_name)
    config = {'host': 'localhost', 'port': args.port}
    es = Elasticsearch([config])
    result = {}
    # telugu is not supported.
    squad_style_dev_data = {'data': [], 'version': 'v2.0'}
    for item in tqdm(input_data):
        topkF1 = []

        # some questions are very long, truncate to 5000 characters
        question = item["question"][:5000]
        lang =  item["language"]
        answer_embeddings = []
        for answer in item["answers"]:
            answer_embeddings.append(embed_sentences([answer]))
        if lang not in LANGS:
            continue
        index_name = args.index_name_prefix + "_" + lang
        res = search_es(es_obj=es, index_name=index_name, question_text=question, n_results=100)
        # result[item["id"]] = {"hits": res["hits"]["hits"], "answers": item["answers"], "has_answer":False, "question": question}
        for hit in res["hits"]["hits"]:
            doc_sentence_tokenized = sentence_tokenize(hit["_source"]["document_text"], lang)
            doc_embedding = embed_sentences(doc_sentence_tokenized)
            over_threshold=False
            for answer_embedding in answer_embeddings:
                if cosine_similar(answer_embedding, doc_embedding):
                    over_threshold = True
                    break
            topkF1.append(int(over_threshold))
            
            
        results[lang].append({
            "F1@1": int(np.sum(topkF1[:1])>0),
            "F1@5": int(np.sum(topkF1[:5])>0),
            "F1@20": int(np.sum(topkF1[:20])>0),
            "F1@50": int(np.sum(topkF1[:50])>0),
            "F1@100": int(np.sum(topkF1[:100])>0),
        })
        overall_results.append({
            "F1@1": int(np.sum(topkF1[:1])>0),
            "F1@5": int(np.sum(topkF1[:5])>0),
            "F1@20": int(np.sum(topkF1[:20])>0),
            "F1@50": int(np.sum(topkF1[:50])>0),
            "F1@100": int(np.sum(topkF1[:100])>0),
        })
            # answers = [{"text": answer, "answer_start": hit["_source"]["document_text"].find(answer)} for answer in item["answers"]]
            # squad_example = {'context': hit["_source"]["document_text"],
            #                 'qas': [{'question': question, 'is_impossible': False,
            #                         'answers': answers,
            #                             'id': item["id"]}]}
            # squad_style_dev_data["data"].append(
            #                     {"title": hit["_source"]["document_title"], 'paragraphs': [squad_example]})

    for lang in results:
        print(f"{lang.upper()} RESULTS")
        aggregate = defaultdict(list)
        for r in results[lang]:
            for k, v in r.items():
                aggregate[k].append(v)

        for k in aggregate:
            results[lang] = aggregate[k]
            print('{}: {} ...'.format(
                k, np.mean(results[lang])))

    print("OVERALL RESULTS")
    aggregate = defaultdict(list)
    for r in overall_results:
        for k, v in r.items():
            aggregate[k].append(v)

    for k in aggregate:
        overall_results = aggregate[k]
        print('{}: {} ...'.format(
            k, np.mean(overall_results)))
    
    # evaluate top 20 accuracy
    # for q_id in result:
    #     hits = result[q_id]["hits"]
    #     answers = result[q_id]["answers"]
    #     for answer in answers:
    #         for hit in hits:
    #             if answer in hit["_source"]["document_text"]:
    #                 result[q_id]["has_answer"] = True
    #                 break

    # with open(args.output_fp, 'w') as outfile:
    #     json.dump(result, outfile)
    # with open("_squad_format" + args.output_fp, 'w') as outfile:
    #     json.dump(squad_style_dev_data, outfile)
    

    # # calc top 20 recall 
    # top_20_accuracy = len([q_id for q_id, item in result.items() if item["has_answer"] is True]) / len(result)
    # # per language performance 
    # per_lang_performance = {}
    # for lang in LANGS:
    #     question_count = len([q_id for q_id, item in result.items() if lang in q_id])
    #     top_20_accuracy_lang = len([q_id for q_id, item in result.items() if item["has_answer"] is True and lang in q_id]) / question_count
    #     per_lang_performance[lang] = top_20_accuracy_lang
    # print(top_20_accuracy)
    # print(per_lang_performance)

if __name__ == '__main__':
    main()