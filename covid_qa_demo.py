

import streamlit as st
import numpy as np
import time
import math
import argparse

import os
import json
import sys
import argparse
import numpy as np
from multiprocessing import Pool as ProcessPool
from functools import partial
from collections import defaultdict
from tqdm import tqdm
import datetime

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


from testDates import checkDate


# dense
import torch
import faiss
from models.bert_retriever import BERTEncoder
from indexes import Extract_Index
from transformers import AutoConfig, AutoTokenizer, AutoModelForQuestionAnswering, default_data_collator, \
                        MarianMTModel, MarianTokenizer
from utils.torch_utils import load_saved, move_to_cuda
from qa.utils_qa import postprocess_qa_predictions
from qa.trainer_qa import QuestionAnsweringTrainer
from sklearn.cluster import KMeans

from annotated_text import annotated_text
from coqa_process.span_heuristic import find_closest_span_match

#parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--retriever-model-name', type=str, default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
parser.add_argument('--retriever-model', type=str)
parser.add_argument('--qa-model-name', type=str, default='ktrapeznikov/biobert_v1.1_pubmed_squad_v2')
parser.add_argument('--qa-model', type=str, default='ktrapeznikov/biobert_v1.1_pubmed_squad_v2')
parser.add_argument('--index-path', type=str)

args = parser.parse_args()

chosen_model_dense = args.retriever_model_name
chosen_model_reader = args.qa_model_name
reader_path = args.qa_model_name
max_answer_length = 30

cuda = torch.device('cuda')
highlight_colors = ["red","blue","orange", "yellow"]
start_highlight = "<span class ='highlight {}'>"
end_highlight = "</span>"

if os.path.isfile("style.css"):
    style_path="style.css"
elif os.path.isfile("../style.css"):
    style_path="../style.css"
else:
    raise FileNotFoundError("Could not find a css style file")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)



@st.cache(allow_output_mutation=True)
def init_dense(index_path):
    print("Initializing dense index...")
    #model_config = AutoConfig.from_pretrained(chosen_model_dense)
    tokenizer = AutoTokenizer.from_pretrained(chosen_model_dense)
    model = BERTEncoder(args.retriever_model_name)
    if args.retriever_model != "":
        model = load_saved(model,args.retriever_model)
    model.to(cuda)
    model.eval()
    embed_path = os.path.join(index_path, "embeds.npy")
    embeds = np.load(embed_path).astype('float32')
    index = Extract_Index(embeds, gpu=True, dimension=768)
    corpus = json.load(open(os.path.join(index_path, 'id2doc.json')))
    print('Done...')
    return model, tokenizer, index, corpus

@st.cache(allow_output_mutation=True)
def init_reader():
    print("Initializing reader...")
    qa_config = AutoConfig.from_pretrained(chosen_model_reader)
    qa_tokenizer = AutoTokenizer.from_pretrained(chosen_model_reader)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(args.qa_model,
        from_tf=bool(".ckpt" in reader_path),
        config=qa_config,
    )

    qa_model.to(cuda)
    qa_model.eval()

    print("Done...")
    return qa_model, qa_tokenizer


translation_model_names = {
    "Spanish":"Helsinki-NLP/opus-mt-es-en",
    "Chinese":"Helsinki-NLP/opus-mt-zh-en"
    }
@st.cache(allow_output_mutation=True)
def init_translator():
    print("initializing translators...")
    mt_tokenizer_es = MarianTokenizer.from_pretrained(translation_model_names["Spanish"])
    mt_model_es = MarianMTModel.from_pretrained(translation_model_names["Spanish"])
    mt_model_es.to(cuda)
    mt_model_es.eval()

    mt_tokenizer_zh = MarianTokenizer.from_pretrained(translation_model_names["Chinese"])
    mt_model_zh = MarianMTModel.from_pretrained(translation_model_names["Chinese"])
    mt_model_zh.to(cuda)
    mt_model_zh.eval()

    print("Done...")
    return mt_model_es, mt_tokenizer_es, mt_model_zh, mt_tokenizer_zh

def find_span(query,text):
    best_span, best_score = find_closest_span_match(
        text, query)
    if best_score >= 0.5:
        return best_span
    return None

peraton_language_mapping = {
    "spa": "Spanish",
    "ger": "German",
    "por": "Portuguese",
    "fre": "French",
    "tur": "Turkish",
    "rus": "Russian",
    "dut": "Dutch",
    "ita": "Italian",
    "pol": "Polish",
    "slv": "Slovene",
    "cze": "Czech",
    "eng": "English",
    "chi": "Chinese"
}

def checkLanguage(doc_language, language_selection):
    if "All" in language_selection:
        return True
    elif peraton_language_mapping[doc_language] in language_selection:
        return True
    else:
        return False

if __name__ =='__main__':

    dense_index_path = args.index_path
    model, tokenizer, index, corpus = init_dense(dense_index_path)
    qa_model, qa_tokenizer = init_reader()
    mt_model_es, mt_tokenizer_es, mt_model_zh, mt_tokenizer_zh = init_translator()
    dateFlag = False
    local_css(style_path)
    analysis = st.sidebar.selectbox('Select number of articles', ['1', '2', '3', '4', '5', '10', '20'])
    analysisInt = int(analysis)
    language_selection = st.sidebar.multiselect('Select one or more article languages', ['All','Chinese','English','Spanish'], default=['All'])
    if "All" in language_selection:
        language_selection=["All"]
    startDate = st.sidebar.date_input('start date', datetime.date.today())
    endDate = st.sidebar.date_input('end date', datetime.date.today())
    if startDate > endDate:
        st.sidebar.error('Invalid! Make sure start date is before end date.')
        dateFlag = True
    else:
        dateFlag = False
    st.markdown(
        """ <style> .sidebar .sidebar-content { background-image: linear-gradient(#2e7bcf,#2e7bcf); color: green; } </style> """,
        unsafe_allow_html=True, )
    st.title("Ask any question about COVID-19!")

    query = st.text_input('Enter your question')

    if query:

        # dense search
        with torch.no_grad():
            q_encodes = tokenizer(
                [query], max_length=50, return_tensors="pt", truncation=True, padding=True)
            batch_q_encodes = move_to_cuda(dict(q_encodes))
            q_embeds = model(
                batch_q_encodes["input_ids"], batch_q_encodes["attention_mask"], batch_q_encodes.get("token_type_ids", None))
            q_embeds_numpy = q_embeds.cpu().contiguous().numpy()


            #retrieve top 2048 (this is the max number allowed) documents for query and filter based on dates selected
            curr_count = 2048
            distances, indices = index.search(curr_count, q_embeds_numpy)
            top_doc_ids = indices[0]
            topk_docs = [{"title": corpus[str(doc_id)][0], "text": corpus[str(doc_id)][1],
                "date": corpus[str(doc_id)][4], "journal": corpus[str(doc_id)][5], 
                "language": corpus[str(doc_id)][7]} for doc_id in top_doc_ids]
            # filter for docs in correct date range
            topk_docs_date = []
            for doc in topk_docs:
                if checkDate(datetime.datetime(startDate.year, startDate.month, startDate.day),
                        datetime.datetime(endDate.year, endDate.month, endDate.day), doc['date']):
                    topk_docs_date.append(doc)

            if len(topk_docs_date) == 0:
                st.warning("No articles found in the date range, retrieving most relevant articles outside of date range.")
                topk_docs_date = topk_docs
            topk_docs = topk_docs_date

            # filter for docs with selected language
            topk_docs_lang = []
            for doc in topk_docs:
                if checkLanguage(doc['language'], language_selection):
                    topk_docs_lang.append(doc)
            if len(topk_docs_lang) == 0:
                st.warning("No matching articles with the selected language(s), retrieving most relevant articles from any language.")
                topk_docs_lang = topk_docs
            topk_docs = topk_docs_lang
            #topk_docs = topk_docs[:topk]




            contexts = []
            answer_contexts = []
            answers = []

            doc_probs = []

            topk_docs = topk_docs[:30]
            non_dup = []
            texts = set()
            for item in topk_docs:
                if item['title'] not in texts:
                    texts.add(item['title'])
                    non_dup.append(item)
            topk_docs = non_dup
            num_docs = float(len(topk_docs))
            corpus = [x['text'] for x in topk_docs]


            # TODO: If we want to cluster documents, need to use a multilingual cluster algorithm (eg. cluster in embedding space)
            #cluster top 30 documents for retrieval diversity and extract top documents from each cluster
            # vectorizer = TfidfVectorizer()
            # X = vectorizer.fit_transform(corpus)
            # num_clusters = min(len(topk_docs),3)
            # kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
            # proportions = {0: 0, 1: 0, 2: 0}
            # for label in kmeans.labels_:
            #     proportions[label] += 1
            # cluster_counts = {}
            # for k,v in proportions.items():
            #     cluster_counts[k] = math.ceil((v/num_docs) * 10)
            # cluster_docs = []
            # for i,doc in enumerate(topk_docs):
            #     label = kmeans.labels_[i]
            #     if cluster_counts[label] > 0:
            #         cluster_counts[label] -= 1
            #         cluster_docs.append(doc)
            #     else:
            #         continue
            # topk_docs = cluster_docs


            for doc in topk_docs:
                if doc["text"] == "":
                    continue
                #run through the qa/reading module for each document for the question
                inputs = qa_tokenizer.encode_plus(query, doc["text"], add_special_tokens=True, return_tensors="pt", truncation=True)
                inputs.to(cuda)
                input_ids = inputs["input_ids"].tolist()[0]

                query_inputs = qa_tokenizer.encode_plus(query, add_special_tokens=True, return_tensors="pt")
                query_input_ids = query_inputs["input_ids"].tolist()[0]
                query_decoded = qa_tokenizer.decode(query_input_ids, skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=True)

                text_tokens = qa_tokenizer.convert_ids_to_tokens(input_ids)
                outputs = qa_model(**inputs)

                start_logits, end_logits = outputs.start_logits.cpu()[0], outputs.end_logits.cpu()[0]

                prelim_predictions = []
                start_indexes = np.argsort(start_logits).tolist()[-1: -60 - 1: -1]
                end_indexes = np.argsort(end_logits).tolist()[-1: -60 - 1: -1]


                #from huggingface utils_qa.py
                #extract start and end tokens predictions that are within context of the input and where start < end position
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                        # to part of the input_ids that are not in the context.
                        if (
                                start_index < len(query_input_ids)
                                or end_index < len(query_input_ids)
                        ):
                            continue
                        # Don't consider answers with a length that is either < 0 or > max_answer_length.
                        if end_index <= start_index or end_index - start_index + 1 > max_answer_length:
                            continue

                        prelim_predictions.append(
                            {
                                "score": start_logits[start_index] + end_logits[end_index],
                                "start_logit": start_logits[start_index],
                                "end_logit": end_logits[end_index],
                                "start_index": start_index,
                                "end_index": end_index,
                            }
                        )
                predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:15]


                answer_start = predictions[0]["start_index"]
                answer_end = predictions[0]["end_index"]
                answer_pairs = [(answer_start, answer_end)]

                # try to get multiple answer spans for the text that do not overlap each other
                all_end_toks = [answer_end]
                all_start_tok = [answer_start]
                for pred in predictions[1:20]:
                    flag = False
                    for start_tok,end_tok in answer_pairs:
                        if (pred["start_index"] > end_tok or pred["end_index"] < start_tok and pred["score"].item() > 0):
                            continue
                        else:
                            flag = True
                            break
                    if flag:
                        continue
                    else:
                        answer_pairs.append((pred["start_index"], pred["end_index"]))

                answer_pairs.sort()
                # answer = qa_tokenizer.decode(input_ids[answer_start:answer_end],skip_special_tokens=True)

                #get positions of start and end tokens in text string
                full_text = qa_tokenizer.decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                answer_tok_indices = []
                for (start_tok,end_tok) in answer_pairs:
                    pair_start_len = len(qa_tokenizer.decode(input_ids[:start_tok], skip_special_tokens=True,
                                            clean_up_tokenization_spaces=True))
                    pair_end_len = len(qa_tokenizer.decode(input_ids[:end_tok], skip_special_tokens=True,
                                                             clean_up_tokenization_spaces=True))
                    while pair_end_len < len(full_text):
                        if full_text[pair_end_len] != " ":
                            pair_end_len += 1
                        else:
                            break
                    answer_tok_indices.append((pair_start_len,pair_end_len))

                #add the highlight commands to the text so that the answer spans are highlighted in red
                curr_tok = 0
                with_highlight = ""
                highlighted_answers = []
                for i, (start_tok,end_tok) in enumerate(answer_tok_indices):
                    highlighted_answers.append(full_text[start_tok:end_tok])
                    with_highlight += full_text[curr_tok:start_tok] + start_highlight.format(highlight_colors[i]) + full_text[start_tok:end_tok] + end_highlight
                    curr_tok = end_tok
                with_highlight += full_text[curr_tok:]

                answer_contexts.append(with_highlight[len(query_decoded):])
                answers.append(highlighted_answers)
                scores = np.array([pred.pop("score") for pred in predictions])
                exp_scores = np.exp(scores - np.max(scores))
                probs = exp_scores / exp_scores.sum()
                doc_probs.append(probs[0])
                contexts.append(doc["text"])


        #reorder documents based on highest answer confidence for each document
        zipped = list(zip(doc_probs,answer_contexts,answers,topk_docs))
        zipped.sort(key=lambda x: x[0])
        zipped.reverse()
        doc_probs, answer_contexts, answers, topk_docs = zip(*zipped)

        #present documents and highlighted answers to users
        if len(topk_docs[:analysisInt]) == 1:
            st.markdown(f'## **Top {analysisInt} Retrieved Article**')
        else:
            st.markdown(f'## **Top {analysisInt} Retrieved Articles**')
        counter = 1
        for count,doc in enumerate(topk_docs[:analysisInt]):
            translated_answers = []
            if doc["language"] == "spa":
                translations = mt_model_es.generate(**mt_tokenizer_es(answers[count], padding=True, return_tensors="pt").to(cuda))
                for i, translation in enumerate(translations):
                    translated_answer = start_highlight.format(highlight_colors[i]) + mt_tokenizer_es.decode(translation, skip_special_tokens=True, clean_up_tokenization_spaces=True) + end_highlight
                    translated_answers.append(translated_answer)
            if doc["language"] == "chi":
                translations = mt_model_zh.generate(**mt_tokenizer_zh(answers[count], padding=True, return_tensors="pt").to(cuda))
                for translation in translations:
                    translated_answer = start_highlight.format(highlight_colors[i]) + mt_tokenizer_zh.decode(translation, skip_special_tokens=True, clean_up_tokenization_spaces=True) + end_highlight
                    translated_answers.append(translated_answer)
            with st.beta_expander("{}, {}".format(doc['journal'], doc['date'])):
                st.markdown('**Title:** {}'.format(doc['title']))
                st.markdown('**Language:** {}'.format(doc['language']))
                st.markdown('**Journal Text**')
                new_text = answer_contexts[count]
                # st.markdown("{}".format(new_text),unsafe_allow_html=True)
                st.text("{}".format(new_text))
                if translated_answers:
                    st.markdown("**English Translation**")
                    for translated_answer in translated_answers:
                        # st.markdown("{}".format(translated_answer,unsafe_allow_html=True))
                        st.text("{}".format(translated_answer))
            counter += 1



