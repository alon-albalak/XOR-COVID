CORD_VERSION=2021-05-31

mkdir CORD19
wget https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/cord-19_$CORD_VERSION.tar.gz
tar -xf cord-19_$CORD_VERSION.tar.gz
rm cord-19_$CORD_VERSION.tar.gz
pushd $CORD_VERSION
tar -xf document_parses.tar.gz

rm changelog cord_19_embeddings.tar.gz document_parses.tar.gz

popd
python3 -m nltk.downloader punkt
python3 splitCORD.py \
    --metadata-file=$CORD_VERSION/metadata.csv \
    --pmc-path=$CORD_VERSION/document_parses/pmc_json \
    --out-path=$CORD_VERSION/CORD19_corpus.jsonl

rm $CORD_VERSION/document_parses $CORD_VERSION/metadata.csv