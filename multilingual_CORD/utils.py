import os
import json

from transformers import MarianMTModel, MarianTokenizer
model_name="Helsinki-NLP/opus-mt-es-en"

def get_titles(path="es_articles"):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    for filename in os.listdir(path):
        if filename.endswith(".json"):
            file = json.load(open(os.path.join(path,filename)))
            title = file["metadata"]["title"]
            print(title)
            # translated = model.generate(**tokenizer(title, return_tensors="pt"))
            # print(tokenizer.decode(translated[0], skip_special_tokens=True))