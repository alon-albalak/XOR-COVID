GPU=$1

# for the moment, pretend we pull it from scratch
# mkdir covid-qa-multilingual
# cd covid-qa-multilingual
# wget https://raw.githubusercontent.com/deepset-ai/COVID-QA/master/data/question-answering/COVID-QA.json

# in reality, we use Sharon's pre-processed covid-QA dataset
CUDA_VISIBLE_DEVICES=$GPU python3 translate_split_covidQA.py