openai.organization = "" #use your openai organization id
openai.api_key = "" #use your openai api key
MODEL = "gpt-3.5-turbo"
DATASET = "dataset/multitude.csv.gz"

import pandas as pd
import numpy as np
import torch
import time
from tqdm import tqdm
import backoff
import openai
from langcodes import *

@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError))
def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)

@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError, openai.error.APIConnectionError, openai.error.ServiceUnavailableError, openai.error.Timeout))
def chat_completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

model_name = MODEL.split("/")[-1]
df = pd.read_csv(DATASET)

subset = df#[:10]
generated = [""] * len(subset)

with torch.no_grad():
  #for index, row in subset.iterrows():
  for index, row in tqdm(subset.iterrows(), total=subset.shape[0]):
    if ("generated" in row.index) and (row['generated'] is not np.NaN) and (str(row['generated']) != "nan"):
      generated[index] = row['generated']
      #print(index, 'skipping')
      continue
    #for testuing purpose
    else:
      #print(index, 'processing')
      #continue
      pass
    
    language_name = Language.make(language=row.language).display_name()
    text = row['text']
    prompt = f'Paraphrase the following text in {language_name} language: {text}'
    if ("gpt-3.5-turbo" in MODEL) or ("gpt-4" in MODEL):
        result = chat_completions_with_backoff(model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=512, top_p=0.95).choices[0].message.content
        time.sleep(2) #to not reach openai rate limit of requests per minute, or use backoff
    generated[index] = result
    subset['generated'] = generated
    subset.to_csv(DATASET.replace('.csv', '_obfuscated_paraphrased-ChatGPT.csv'), index=False)
