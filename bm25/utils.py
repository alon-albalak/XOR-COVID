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
            wiki_id = obj["id"]
            title = obj["title"]
            title_id = make_wiki_id(title, 0)
            text_with_links = obj["text"]

            hyper_linked_titles_text = ""
            # When we consider the whole article as a document unit (e.g., SQuAD Open, Natural Questions Open)
            # we'll keep the links with the original articles, and dynamically process and extract the links
            # when we process with our selector.
            extracted_items.append({"wiki_id": wiki_id, "title": title_id,
                                    "plain_text": text_with_links,
                                    "hyper_linked_titles": hyper_linked_titles_text,
                                    "original_title": title})

    return extracted_items