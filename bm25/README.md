# BM25 baseline

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

