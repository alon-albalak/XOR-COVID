# BM25 baseline

### The code found here has been adopted from [this repo by Akari Asai](https://github.com/AkariAsai/XORQA/tree/main/baselines/bm25)

## Install and set up Elastic Search

### Create new environment and install requirements
```bash
python3 -m venv bm25
source bm25/bin/activate
pip3 install -r bm25_requirements.txt
python3 -m nltk.downloader "punkt"
```
### Elastic search installation
`bash install_elasticsearch.sh`

### To start/stop an ES instance:
`bash server.sh [start|stop]`

## Index documents and search

1. Create db from preprocessed jsonlines file

`python3 build_db.py /path/to/preprocessed/data/file.jsonl /path/to/output/filename.db`

2. Index documents (all languages)

`python3 build_es.py --db_path=/path/to/your/db.db --config_folder=/path/to/configs --port=9200 --index_prefix=/your/index/name`

3. Search documents based on BM25 score

This function will translate all questions into all languages in the databases and aggregate BM25 search results across all languages

`python3 es_search_multi.py --index_prefix=/your/index/name --input_data_file_name=/path/to/your/retrieval/data.txt --port=9200`