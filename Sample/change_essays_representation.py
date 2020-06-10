"""
Originally the essays are distributed in individual files,
with metaphorical words starting with 'M_'
This file reads the texts from the essays dataset.
Finds the metaphors marked in the text.
Correct the misspelled words
It then produces representation of: text, metaphor, offset
"""

import pandas as pd
import os
from Sample.modules.kmeans_abs_ratings_cosine_edit_distance import get_abstractness_rating
from textblob import TextBlob

DATA_PATH = "/home/aman/IITC/Research_Project/Sample/data/toefl_sharedtask_dataset/essays"

texts = []
header = ['text', 'metaphor', 'offset']
df = pd.DataFrame(columns=header)

abstractness_rating_dict = get_abstractness_rating()


def clean_text(text):
    metaphors = []
    offsets = []
    tokens = text.strip().split()
    puncts = ['.', ',', '!', '?', ':', '\'']
    tokens = [token for token in tokens if token not in puncts]

    for i, token in enumerate(tokens):
        if token.startswith("M_"):
            metaphors.append(token[2:])
            offsets.append(i)
            tokens[i] = token.replace(token, token[2:])

        elif token not in abstractness_rating_dict.keys():
            print(f'Misspelled word found: {token}')
            tokens[i] = str(TextBlob(token).correct())
            print(f'Replaced with: {tokens[i]}')

    new_text = ' '.join(tokens)
    return new_text, metaphors, offsets


for essay in os.listdir(DATA_PATH):
    path = os.path.join(DATA_PATH, essay)
    with open(path, 'r', encoding='utf8') as textfile:
        lines = textfile.readlines()
        for line in lines:
            text, metaphors, offsets = clean_text(line)
            for i, metaphor in enumerate(metaphors):
                row = [text, metaphor, offsets[i]]
                df = df.append(pd.DataFrame([row], columns=header), ignore_index=True)

df.to_csv(os.path.join(DATA_PATH, '../essays.csv'))
