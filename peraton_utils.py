import json
import nltk


def get_language_data(path_prefix="peratonCovidNonEnglish/"):
    files = [
        path_prefix + "covid0000.json",
        path_prefix + "covid0001.json",
        path_prefix + "covid0002.json",
        path_prefix + "covid0003.json",
        path_prefix + "covid0004.json",
        path_prefix + "covid0005.json",
    ]

    total = 0
    langs = {}
    for file in files:
        tmp = json.load(open(file))
        for uuid, data in tmp.items():
            if "OtherAbstractLanguages" in data.keys():
                for lang in data["OtherAbstractLanguages"]:
                    if lang not in langs:
                        langs[lang] = 0
                    langs[lang] += 1
            total += 1

    print(f"Datums per language: {langs}")
    print(f"Total non-engligh abstracts: {sum(langs.values())}")
    print(f"Total abstracts: {total}")


def chunk_text(text, lang="eng"):
    langs = {
        "spa": "spanish",
        "ger": "german",
        "por": "portuguese",
        "fre": "french",
        # "hun": "hungarian", doesn't exist
        "tur": "turkish",
        "rus": "russian",
        "dut": "dutch",
        "ita": "italian",
        "pol": "polish",
        "slv": "slovene",
        "cze": "czech",
    }
    if lang == "eng":
        sentences = nltk.sent_tokenize(text)
    elif lang in langs:
        tokenizer = nltk.data.load(f"tokenizers/punkt/{langs[lang]}.pickle")
        sentences = tokenizer.tokenize(text)
    else:
        if lang == "chi":
            sentences = [sent + "\u3002" for sent in text.split("\u3002") if len(sent) > 3]
        elif lang == "ara":
            sentences = [sent + "." for sent in text.split(".") if len(sent) > 3]  # probably not great
        elif lang == "kor":
            sentences = [sent + "." for sent in text.split(".") if len(sent) > 3]  # probably not great
        else:
            return []
    chunked = []
    curr_len = 0
    curr_sentences = []
    # split into semi-even chunks
    for sent in sentences:
        if curr_len + len(sent.split()) > 90:
            curr_sentences.append(sent)
            chunked.append(" ".join(curr_sentences))
            curr_len = 0
            curr_sentences = []
        else:
            curr_len += len(sent.split())
            curr_sentences.append(sent)
    if curr_len > 90 or len(chunked) == 0:
        chunked.append(" ".join(curr_sentences))
    else:
        last = chunked.pop()
        chunked.append(" ".join([last] + curr_sentences))
    return chunked


def convert_peraton_jsons_to_CORD_jsonl(
    dumpPath="peraton_corpus.jsonl", path_prefix="XOR-COVID/peratonCovidNonEnglish/", include_english=True
):
    """
    Converts the data from peraton into a single jsonlines file where each line is a dict:
            {
                id,
                title,
                text,
                index,
                date,
                journal,
                authors,
                language
            }
    """

    files = [
        path_prefix + "covid0000.json",
        path_prefix + "covid0001.json",
        path_prefix + "covid0002.json",
        path_prefix + "covid0003.json",
        path_prefix + "covid0004.json",
        path_prefix + "covid0005.json",
    ]

    index = 0

    dumpFile = open(dumpPath, "w", encoding="utf8")
    for file in files:
        json_file = json.load(open(file))
        for uuid, data in json_file.items():
            # Make sure we have a publication date
            date_preference = ["PUBLISHED", "REVISED", "ELECTRONIC_VERSION", "COMPLETED"]
            date = ""
            for pref in date_preference:
                if f"{pref}_DATE" in data.keys():
                    date = data[f"{pref}_DATE"]
                    break
            if date == "":
                continue

            # make sure there are authors
            if "Authors" in data.keys():
                authors = data["Authors"]
            else:
                authors = "unknown"

            # make sure there is a title
            if "TITLE" in data.keys():
                title = data["TITLE"]
            else:
                title = "unknown"

            if not include_english:
                pass
            else:
                text = " ".join(data["CONTEXTS"])
                chunked = chunk_text(text)
                for chunkedParagraph in chunked:
                    entry = {
                        "id": data["PMID"],
                        "title": title,
                        "text": chunkedParagraph,
                        "index": index,
                        "date": date,
                        "journal": "unknown",  # we currently don't have this information
                        "authors": authors,
                        "language":"eng"
                    }
                    json.dump(entry, dumpFile, ensure_ascii=False)
                    dumpFile.write("\n")
                    index += 1

            if "OtherAbstractLanguages" in data.keys():
                for lang, text in zip(data["OtherAbstractLanguages"], data["OtherAbstracts"]):
                    article_text = " ".join(text["text"])
                    chunked = chunk_text(article_text, lang=lang)
                    for chunkedParagraph in chunked:
                        entry = {
                            "id": data["PMID"],
                            "title": title,
                            "text": chunkedParagraph,
                            "index": index,
                            "date": date,
                            "journal": "unknown",  # we currently don't have this information
                            "authors": authors,
                            "language":lang
                        }
                        json.dump(entry, dumpFile, ensure_ascii=False)
                        dumpFile.write("\n")
                        index += 1
    dumpFile.close()