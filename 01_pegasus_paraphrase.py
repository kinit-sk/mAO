import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm
from polyglot.text import Text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = PegasusTokenizer.from_pretrained('tuner007/pegasus_paraphrase')
model = PegasusForConditionalGeneration.from_pretrained('tuner007/pegasus_paraphrase').to(device)

def get_response(input_text,num_return_sequences,num_beams):
  batch = tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(device)
  translated = model.generate(**batch, max_length=60, num_beams=num_beams, num_return_sequences=num_return_sequences, do_sample=True, temperature=1.5)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text

def get_paraphrase(text, lang):
  sentences = Text(text, hint_language_code=lang).sentences
  sentences = [str(x) for x in sentences]
  paraphrase = []
  for sentence in sentences:
    temp = get_response(sentence, 1, 10)
    paraphrase.append(temp)
  paraphrase = [' '.join(x) for x in paraphrase]
  paraphrase = ' '.join(paraphrase)
  return paraphrase
  
#lang = 'en'
#text = "Artificial Intelligence (AI) and Machine Learning (ML) are two closely related but distinct fields within the broader field of computer science. AI is a discipline that focuses on creating intelligent machines that can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and natural language processing. It involves the development of algorithms and systems that can reason, learn, and make decisions based on input data."
#print(get_paraphrase(text, lang))

test = pd.read_csv('dataset/multitude.csv.gz')
subset = test#[:100]

generated = [""] * len(subset)
model = model.eval()

with torch.no_grad():
  for index, row in tqdm(subset.iterrows(), total=subset.shape[0]):
    if ("generated" in row.index) and (row['generated'] is not np.NaN) and (str(row['generated']) != "nan"):
      generated[index] = row['generated']
      continue
    generated[index] = get_paraphrase(row.text, row.language)
    subset['generated'] = generated
    subset.to_csv(f'temp.csv', index=False)

print(pd.DataFrame([subset['text'] == subset['generated']]).T.value_counts())

subset.to_csv('dataset/multitude_obfuscated_pegasus-paraphrase.csv.gz', index=False)
