#download COUGH dataset
wget https://github.com/sunlab-osu/covid-faq/archive/refs/heads/main.tar.gz

mkdir COUGH
tar -xzf main.tar.gz -C COUGH

python3 splitCOUGH --save_path=COUGH --data_path=COUGH/covid-faq-main/data/FAQ_Bank_eval.csv
rm -r COUGH/covid-faq-main
