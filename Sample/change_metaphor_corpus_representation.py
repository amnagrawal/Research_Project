import pandas as pd
import os
from Sample.modules.kmeans_abs_ratings_cosine_edit_distance import get_abstractness_rating
from textblob import TextBlob


DATA_PATH = "/home/aman/IITC/Research_Project/Sample/data"

data = pd.read_csv(os.path.join(DATA_PATH, 'metaphor-corpus.csv'))
header = ['text', 'source', 'target']
df = pd.DataFrame(columns=header)

abstractness_rating_dict = get_abstractness_rating()

for index, row in data.iterrows():
    source = row['Source LM']
    text = row['Sentence']
    target = row['Target LM']
    tokens = text.split(' ')

    offset = 0
    for i, token in enumerate(tokens):
        if token not in abstractness_rating_dict.keys():
            tokens[i] = str(TextBlob(token).correct())

    new_text = ' '.join(tokens)
    row = [new_text, source, target]
    df = df.append(pd.DataFrame([row], columns=header), ignore_index=True)

df.to_csv(os.path.join(DATA_PATH, 'new_metaphor_corpus.csv'))
