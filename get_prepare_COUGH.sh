GPU=$1

#download COUGH dataset
wget https://github.com/sunlab-osu/covid-faq/archive/refs/heads/main.tar.gz

mkdir COUGH
tar -xzf main.tar.gz -C COUGH
rm main.tar.gz

python3 splitCOUGH.py --save_path=COUGH --data_path=COUGH/covid-faq-main/data/FAQ_Bank.csv
rm -r COUGH/covid-faq-main

# translate the english answers into other languages
#   and translate other language questions into english
CUDA_VISIBLE_DEVICES=$GPU python3 translate_COUGH.py --data_path=COUGH/retrieval_test.txt --save_path=COUGH/en2all_test.txt
CUDA_VISIBLE_DEVICES=$GPU python3 translate_COUGH.py --data_path=COUGH/retrieval_dev.txt --save_path=COUGH/en2all_dev.txt
CUDA_VISIBLE_DEVICES=$GPU python3 translate_COUGH.py --data_path=COUGH/retrieval_train.txt --save_path=COUGH/en2all_train.txt