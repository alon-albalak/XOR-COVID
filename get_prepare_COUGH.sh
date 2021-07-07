#download COUGH dataset
wget https://github.com/sunlab-osu/covid-faq/archive/refs/heads/main.tar.gz

mkdir COUGH
tar -xzf main.tar.gz -C COUGH
rm main.tar.gz

python3 splitCOUGH.py --save_path=COUGH --data_path=COUGH/covid-faq-main/data/FAQ_Bank.csv
rm -r COUGH/covid-faq-main
