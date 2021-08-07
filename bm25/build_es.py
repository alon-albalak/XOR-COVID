from elasticsearch import Elasticsearch
import argparse
import os
from doc_db import DocDB
import re
import json
from tqdm import tqdm
from utils import peraton_lang_id_to_ISO6391

def build_all_index(index_prefix, db_path, config_folder, port=9200):
    db = DocDB(db_path)

    # initialize the elastic search
    config = {'host': 'localhost', 'port': port}
    es = Elasticsearch([config])
    for iso_lang in peraton_lang_id_to_ISO6391.values():

        index_name = "{0}_{1}".format(index_prefix, iso_lang)
        if es.indices.exists(index=index_name):
            es.indices.delete(index=index_name)

        index_settings = json.load(open(os.path.join(config_folder,f"{iso_lang}_config.json")))

        es.indices.create(index=index_name, body={"settings": index_settings["settings"]})
    
    # populate index
    # load DB and index in Elastic Search
    es.ping()
    doc_ids = db.get_doc_ids()
    counts = {lang:0 for lang in peraton_lang_id_to_ISO6391.keys()}
    for doc_id in tqdm(doc_ids):
        # if using full journal articles, we can split them into sections (introduction, methods, etc.)
        # sections_paras = db.get_doc_text_section_separations(doc_id)
        # for section in sections_paras:
        #     section_name = section["section_name"]
        #     parent_section_name = section["parent_section_name"]
        #     paragraphs = section["paragraphs"]
        #     title = doc_id.split("_0")[0]
        #     for para_idx, para in enumerate(paragraphs):
        #         para_title_id = "title:{0}_parentSection:{1}_sectionName:{2}_sectionIndex:{3}".format(title, parent_section_name, section_name, para_idx)
        #         rec = {"document_text": para, "document_title": para_title_id}
        #         try:
        #             index_status = es.index(index=index_name, id=count, body=rec)
        #             count += 1
        #         except:
        #             print(f'Unable to load document {para_title_id}.')

        # when we don't need to split our data into sections, use this:
        doc_title, doc_text, doc_pmid, doc_lang = db.get_doc_info(doc_id)
        if doc_lang in peraton_lang_id_to_ISO6391.keys():
            index_name = "{0}_{1}".format(index_prefix, peraton_lang_id_to_ISO6391[doc_lang])
            rec = {"document_text":doc_text, "document_title":doc_title, "document_pmid":doc_pmid}
            try:
                index_status = es.index(index=index_name, id=counts[doc_lang], body=rec)
                counts[doc_lang] += 1
            except:
                print(f"Unable to load document {doc_pmid}")



    n_records = es.count(index=index_name)['count']
    print(f'Successfully loaded {n_records} into {index_name}')
    return es

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', type=str, required=True,
                        help='Path to sqlite db holding document texts')
    parser.add_argument('--config_folder', type=str, required=True,
                        help='path to the config folder')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--index_prefix', type=str, required=True)

    args = parser.parse_args()

    # args.db_path="bm25/test_db.db"
    # args.config_folder="bm25/es_configs"
    # args.port = 9200
    # args.index_prefix="test_index"


    es = build_all_index(args.index_prefix, args.db_path, args.config_folder, port=args.port)
    question_text = "What did Ron Paul majored in college?"
    res = search_es(es_obj=es, index_name="{0}_en".format(args.index_prefix), question_text=question_text, n_results=5)
    print(res)

    question_text = "¿Cómo se transmite COVID-19 en lugares públicos?"
    res = search_es(es_obj=es, index_name=f"{args.index_prefix}_es", question_text=question_text)
    print(res)

    question_text = "COVID-19 如何在公共场所传播？"
    res = search_es(es_obj=es, index_name=f"{args.index_prefix}_zh", question_text=question_text)
    print(res)

if __name__ == '__main__':
    main()