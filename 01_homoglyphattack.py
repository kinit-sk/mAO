import pandas as pd
import numpy as np
from tqdm import tqdm
from confusables import confusable_characters
import random
random.seed(42)

PROBABILITY = 0.1

random.random() < PROBABILITY

def homoglyph(input_text):
    output = ""
    for char in input_text:
      if (random.random() < PROBABILITY):
        output += confusable_characters(char)[int(random.random()*len(confusable_characters(char)))]
      else:
        output += char
    return output

test = pd.read_csv('dataset/multitude.csv.gz')

subset = test#[:1000]

generated = [""] * len(subset)

for index, row in tqdm(subset.iterrows(), total=subset.shape[0]):
    if ("generated" in row.index) and (row['generated'] is not np.NaN) and (str(row['generated']) != "nan"):
      generated[index] = row['generated']
      continue
    n_try = 0
    while generated[index] == "" or generated[index] == row['text']:
      generated[index] = homoglyph(row['text'])
      n_try += 1
      if n_try >= 10: break
    subset['generated'] = generated
    #subset.to_csv(f'temp.csv', index=False)

subset.to_csv('dataset/multitude_obfuscated_HomoglyphAttack.csv.gz', index=False)
