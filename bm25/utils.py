import unicodedata
import jsonlines


def normalize(text):
    """Resolve different type of unicode encodings / capitarization in HotpotQA data."""
    text = unicodedata.normalize('NFD', text)
    return text[0].capitalize() + text[1:]

def make_wiki_id(title, para_index):
    title_id = "{0}_{1}".format(normalize(title), para_index)
    return title_id

def process_jsonlines(filename):
    """
    This is process_jsonlines method for extracted Wikipedia file.
    After extracting items by using Wikiextractor (with `--json` and `--links` options), 
    you will get the files named with wiki_xx, where each line contains the information of each article.
    e.g., 
    {"id": "316", "url": "https://en.wikipedia.org/wiki?curid=316", "title": "Academy Award for Best Production Design", 
    "text": "Academy Award for Best Production Design\n\nThe <a href=\"Academy%20Awards\">Academy Award</a> for 
    Best Production Design recognizes achievement for <a href=\"art%20direction\">art direction</a> \n\n"}
    This function takes these input and extract items.
    Each article contains one or more than one paragraphs, and each paragraphs are separeated by \n\n.
    """
    # item should be nested list
    extracted_items = []
    with jsonlines.open(filename) as reader:
        for obj in reader:
            index = obj["index"]
            title = obj["title"]
            text = obj["text"]
            pmid = obj["id"]
            date = obj["date"]
            lang = obj["language"]
            
            extracted_items.append({
                "id": index,
                "title":title,
                "text": text,
                "pmid": pmid,
                "date": date,
                "language":lang
                })

    return extracted_items