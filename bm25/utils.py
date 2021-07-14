import unicodedata
import jsonlines

# This includes all supported languages in the peraton covid dataset
peraton_lang_id_to_ISO6391 = {
    "ara":"ar",
    "ger":"de",
    "eng":"en",
    "spa":"es",
    "fre":"fr",
    "hun":"hu",
    "ita":"it",
    "kor":"ko",
    "dut":"nl",
    # "pol":"pl", not working for unknown reason
    "por":"pt",
    "rus":"ru",
    "tur":"tr",
    "chi":"zh",
}

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