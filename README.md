# Multi- and Cross-lingual Open-Domain Question Answering for COVID-19 (Phase B)

This repository contains the source code for an end-to-end open-domain multi-lingual question answering system. The system is made up of two components: a retriever model and a reading comprehension (question answering) model. We provide the code for these two models in addition to demo code based on Streamlit. 


## Installation
To set up the training environment, follow the instructions in requirements_description.txt

Our system uses XLM-RoBERTa, a neural language model that is pretrained on 2.5TB of data in 100 different languages. We use the version found in the 🤗 Transformers library.


## Datasets
- We use the [COUGH](https://github.com/sunlab-osu/covid-faq/) dataset for training our retriever. In addition to the original data, we also include a script to translate some of the data using [MarianMT](https://marian-nmt.github.io/), which translates the answers from english QA pairs into foreign languages, and translates the question from foreign language QA pairs into english. In this way, we create artificial cross-lingual data where the question is in english, but the answer may be in any language.
- We provide the [COVID-QA](https://www.aclweb.org/anthology/2020.nlpcovid19-acl.18.pdf) dataset under the /data directory as well as a script to translate the data using MarianMT machine translation models.
- Additionally, we use an internal version of the CORD-19 dataset which contains article abstracts in english and other languages. This dataset is found in the /internalCORD directory.

To access the COUGH and COVID-QA datasets, we provide simple scripts to download, pre-process, and translate the data. In order to run machine translation, you will need a GPU. The below scripts will save the COUGH data in the /COUGH directory, and the COVID-QA data in the /multilingual_covidQA directory:
```
GPU=0
bash get_prepare_COUGH.sh $GPU
bash get_prepare_covidQA.sh $GPU
```

We maintain the internal CORD dataset in this repo, it is not available publicly. The dataset is stored in the jsonlines format where each line contains a json object with:
  * id: article PMC id
  * title: article title
  * text: article text
  * index: text's index in the corpus (also the same as line number in the jsonl file)
  * date: article date 
  * journal: journal published
  * authors: author list

## Dense Retrieval Model
Now that we have all of our data, we start by training the retrieval model. During training, we use positive and negative paragraphs, positive being paragraphs that contain the answer to a question, and negative ones not. We train on the COUGH dataset (see the Datasets section for more information on COUGH). We have a unified encoder for both questions and text paragraphs that learns to encode questions and associated texts into similar vectors. Afterwards, we use the model to encode the internal CORD-19 corpus. For the retriever, we use the pre-trained XLM-RoBERTa model.

### Training
To train the dense retrieval model, use:
```bash
CUDA_VISIBLE_DEVICES=0 python3 train_retrieval.py \
  --do_train \
  --prefix=COUGH_en2all \
  --predict_batch_size=500 \
  --model_name=xlm-roberta-base \
  --train_batch_size=30 \
  --accumulate_gradients=2 \
  --learning_rate=1e-5 \
  --fp16 \
  --train_file=COUGH/en2all_train.txt \
  --predict_file=COUGH/en2all_dev.txt \
  --seed=16 \
  --eval_period=5000 \
  --max_c_len=300 \
  --max_q_len=30 \
  --warmup_ratio=0.1 \
  --num_train_epochs=20 \
  --output_dir=path/to/model/output
```

Here are things to keep in mind:
```
1. The output_dir flag is where the model will be saved.
2. You can define the init_checkpoint flag to continue fine-tuning on another dataset.
```

### Corpus
Next, to index the internal CORD dataset with the dense retrieval model trained above, use
```
CUDA_VISIBLE_DEVICES=0 python3 encode_corpus.py \
    --do_predict \
    --predict_batch_size 1000 \
    --model_name xlm-roberta-base \
    --fp16 \
    --predict_file /path/to/corpus \
    --max_c_len 300 \
    --init_checkpoint /path/to/saved/model/checkpoint_best.pt \
    --save_path /path/to/encoded/corpus
```

Here are things to keep in mind:
```
1. The predict_file flag should take in your CORD-19 dataset path. It should be a .jsonl file.
2. Look at your output_dir path when you ran train_retrieval.py. After training our model, we should now have a checkpoint in that folder. Copy the exact path onto the init_checkpoint flag here.
3. As previously mentioned, the result of these commands is the corpus (internal CORD) embeddings become indexed. The embeddings are saved in the save_path flag argument. Create that directory path as you wish.
```

### Evaluation
We evaluate retrieval on the test set from the COUGH dataset. We determine the percentage of questions that have retrieved paragraphs with the correct answer across different top-k settings. To determine whether tha answer is correct, we use a pre-trained multilingual Siamese BERT for fuzzy matching and F1 score. The model we use (paraphrase-multilingual-mpnet-base-v2) is publicly available in the [sentence-transformers package](https://www.sbert.net/docs/pretrained_models.html).

To evaluate the retrieval model, use:
```bash
CUDA_VISIBLE_DEVICES=0 python3 eval_retrieval.py \
  --raw_data COUGH/en2all_test.txt \
  --encode_corpus /path/to/encoded/corpus \
  --model_path /path/to/saved/model/checkpoint_best.pt \
  --batch_size 1000 \
  --model_name xlm-roberta-base \
  --topk 100 \
  --dimension 768
```

Here are things to think about:
```
1. The first, second, and third arguments are our COUGH test set, corpus indexed embeddings, and retrieval model respectively.
2. The other flag that is important is topk. This flag determines the quantity of retrieved CORD19 paragraphs.
```

## Reading Comprehension
We utilize a modified version of HuggingFace's question answering scripts to train and evaluate our reading comprehension model. The modifications allow for the extraction of multiple answer spans per document. For reading comprehension, we use an XLM-RoBERTa model which has been pre-trained on the XQuAD dataset.

### Training
To train the reading comprehension model, use:
```bash
CUDA_VISIBLE_DEVICES=0 python3 qa/run_qa.py \
  --model_name_or_path=alon-albalak/xlm-roberta-base-xquad \
  --train_file=multilingual_covidQA/one2one_en2many/qa_train.json \
  --validation_file=multilingual_covidQA/one2one_en2many/qa_dev.json \
  --test_file=multilingual_covidQA/one2one_en2many/qa_test.json \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size=40 \
  --fp16 \
  --learning_rate=3e-5 \
  --num_train_epochs=10 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --evaluation_strategy=epoch \
  --save_strategy=epoch \
  --save_total=1 \
  --logging_strategy=epoch \
  --load_best_model_at_end \
  --metric_for_best_model=f1 \
  --gradient_accumulation_steps=1 \
  --output_dir=/path/to/model/output

```

## Demo
We combine the retrieval and reading models for an end-to-end open-domain question answering demo with Streamlit. This can be run with:
```bash
CUDA_VISIBLE_DEVICES=0 streamlit run covid_qa_demo.py -- \
  --retriever_model_name=xlm-roberta-base \
  --retriever_model=/path/to/saved/retriever_model/checkpoint_best.pt \
  --qa_model_name=alon-albalak/xlm-roberta-base-xquad \
  --qa_model=/path/to/saved/qa_model/ \
  --index_path=/path/to/encoded/corpus
```
Here are things to keep in mind:
```
1. retriever_model is the checkpoint file of your trained retriever model.
2. qa_model is the path to your trained reading comprehension model.
3. index_path is the path to the encoded corpus embeddings.
```
