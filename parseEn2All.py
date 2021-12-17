import os
import json
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from scipy import spatial
import jsonlines

model = SentenceTransformer('sentence-transformers/LaBSE')

in_file = "/path/to/infile"
out_file = "/path/to/outfile"


data = []
ids = []
questionPairs = {}
answerPairs = {}
pairs = {}
with open(infile) as f:
    lines = f.readlines()
    for line in lines:
        sample = json.loads(line)
        data.append(sample)
        if sample['id'] not in pairs:
            pairs[sample['id']] = [sample]
        else:
            pairs[sample['id']].append(sample)

    finalDataset = []
    origPairs = {}
    count = 0
    toDelete = []
    for k,v in pairs.items():
        if len(v) == 1:
            toDelete.append(k)
            origPairs[k] = v
            finalDataset.append(v[0])
        if len(v) != 1:
            for val in v:
                if '|' not in val['language'] and val['language'] != 'en':
                    finalDataset.append(val)
    for k in toDelete:
        del pairs[k]

    originalEn = {}
    count = 0
    forAnswer = []
    for k, v in pairs.items():
        if len(v) == 9:
            forAnswer.append(k)
    for k in forAnswer:
        answerPairs[k] = pairs[k]
        for v in answerPairs[k]:
            if v['language'] == 'en':
                originalEn[k] = v


    count = 0
    lang = []
    for k,v in answerPairs.items():
        finalDataset.append(originalEn[k])
        origEmbed = model.encode(originalEn[k]['answers'][0])
        for val in v:
            if val['language'] != 'en':
                embeddings = model.encode(val['answers'][0])
                result = 1 - spatial.distance.cosine(origEmbed, embeddings)
                if result > 0.9:
                    finalDataset.append(val)

    with jsonlines.open(outfile, "w") as writer:
        writer.write_all(finalDataset)





