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
# modelTransf = SentenceTransformer("bert-base-nli-mean-tokens")

from utils import peraton_lang_id_to_ISO6391
LANGS=peraton_lang_id_to_ISO6391.values()
translated_langs = ["ar","de","en","es","fr","ru","zh"]

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
    parser.add_argument('--index_prefix', type=str, help='Path to index')
    parser.add_argument('--input_data_file_name', type=str)
    parser.add_argument('--port', type=int)

    args = parser.parse_args()

    # args.index_prefix="peraton_db"
    # args.input_data_file_name="COUGH/retrieval_all_langs_dev.txt"
    # args.port=9200

    results = {lang:[] for lang in LANGS}
    overall_results = []

    input_data = read_jsonlines(args.input_data_file_name)
    config = {'host': 'localhost', 'port': args.port}
    es = Elasticsearch([config])
    result = {}
    # telugu is not supported.
    for item in tqdm(input_data):

        topkF1 = []
        aggregated_results = []
        original_lang =  item["original_lang"]

        
        for lang in translated_langs:
            if f"question_{lang}" in item.keys():
                index_name = args.index_prefix + "_" + lang
                # some questions are very long, truncate to 5000 characters
                question = item[f"question_{lang}"][:5000]
                res = search_es(es_obj=es, index_name=index_name, question_text=question, n_results=100)
                aggregated_results.extend(res['hits']['hits'])
                
        aggregated_results = sorted(aggregated_results, key=lambda hit: hit['_score'], reverse=True)[:100]


        answer_embeddings = []
        for answer in item["answers"]:
            answer_embeddings.append(embed_sentences([answer]))


        for hit in aggregated_results:
            doc_lang = hit['_index'][-2:]
            doc_sentence_tokenized = sentence_tokenize(hit["_source"]["document_text"], doc_lang)
            doc_embedding = embed_sentences(doc_sentence_tokenized)
            over_threshold=False
            for answer_embedding in answer_embeddings:
                if cosine_similar(answer_embedding, doc_embedding):
                    over_threshold = True
                    break
            topkF1.append(int(over_threshold))
            
        overall_results.append({
            "F1@1": int(np.sum(topkF1[:1])>0),
            "F1@5": int(np.sum(topkF1[:5])>0),
            "F1@20": int(np.sum(topkF1[:20])>0),
            "F1@50": int(np.sum(topkF1[:50])>0),
            "F1@100": int(np.sum(topkF1[:100])>0),
        })

    print("OVERALL RESULTS")
    aggregate = defaultdict(list)
    for r in overall_results:
        for k, v in r.items():
            aggregate[k].append(v)

    for k in aggregate:
        overall_results = aggregate[k]
        print('{}: {} ...'.format(
            k, np.mean(overall_results)))
    
if __name__ == '__main__':
    main()