DATAPATH = "./dataset/"
MODELPATH = "./models/"
CACHE = "./cache/"
offload_folder = "./offload_folder/"

import sys

PRE_TRAINED_MODEL_NAME = sys.argv[1] #model name/identifier or path
model_name = PRE_TRAINED_MODEL_NAME.split('/')[-1]
dataset = sys.argv[2] #'en', 'es', 'ru', 'all', 'en3'
generative_model = sys.argv[3] #'text-davinci-003', 'gpt-3.5-turbo', 'gpt-4', 'llama-65b', 'opt-66b', 'opt-iml-max-1.3b', 'all'
output_model = f'{MODELPATH}{model_name}-finetuned-{dataset}-{generative_model}'
if dataset == "no" or generative_model == "no":
  output_model = PRE_TRAINED_MODEL_NAME #e.g. for testing of pretrained models such as openai-detector
if dataset == "inname" or generative_model == "inname":
  output_model = MODELPATH + PRE_TRAINED_MODEL_NAME #e.g. for testing of finetuned models by their name

obfuscated_dataset = sys.argv[4] #e.g. multitude_obfuscated_dipper.csv
obfuscated_dataset = obfuscated_dataset.split('/')[-1]

import os
os.environ['HF_HOME'] = CACHE

import pandas as pd
from sklearn.metrics import classification_report
from transformers import pipeline
import numpy as np
import torch
import gc
import nvidia_smi, psutil, shutil
import time
from tqdm import tqdm
#from polyglot.text import Text

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

def report_gpu():
  if('cpu' in device.type): return
  nvidia_smi.nvmlInit()
  handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
  info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
  print("GPU [GB]:", f'{info.used/1024/1024/1024:.2f}', "/", f'{info.total/1024/1024/1024:.1f}')
  nvidia_smi.nvmlShutdown()
  print('RAM [GB]:', f'{psutil.virtual_memory()[3]/1024/1024/1024:.2f}', "/", f'{psutil.virtual_memory()[0]/1024/1024/1024:.1f}')

start = time.time()
if "ruroberta-ruatd-binary" in output_model:
  classifier = pipeline("text-classification", model=output_model, tokenizer="ai-forever/ruRoberta-large", device=device)#, torch_dtype=torch.float16)
else:
  classifier = pipeline("text-classification", model=output_model, device=device)#, torch_dtype=torch.float16)
end = time.time()
print(f"{output_model.split('/')[-1]} loading took {(end - start)/60} min")
try:
  print(f"{output_model.split('/')[-1]} memory footprint {classifier.model.get_memory_footprint()/1024/1024/1024} GB")
except:
  pass
try:
  report_gpu()
except:
  pass

#remove unfinished final sentence from text
def remove_unended_sentence(text_string):
  #text = Text(text_string)
  #if (len(text.sentences) > 1):
  #  if (text.sentences[-1].words[-1] not in ['。', '؟', '!', '?', '.']): #final sentence not ended by any of these characters
  #    return text_string.removesuffix(str(text.sentences[-1]))
  idx = max(text_string.rfind('。'), text_string.rfind('؟'), text_string.rfind('!'), text_string.rfind('?'), text_string.rfind('.'))
  if idx > -1:
    return text_string[:idx]
  return text_string

def predict(df):
  preds = ['unknown'] * len(df)
  scores = [0] * len(df)
  for index, row in tqdm(df.iterrows(), total=len(df)):
    text = row['text']
    #text = remove_unended_sentence(text)
    tokenizer_kwargs = {'truncation':True,'max_length':512}
    pred = classifier(text, **tokenizer_kwargs)
    preds[index] = pred[0]['label']
    scores[index] = pred[0]['score']
  return preds, scores
    
test = pd.read_csv(DATAPATH + obfuscated_dataset)
if "generated" not in test.columns:
  test["generated"] = test["text"]
test = test[test.split == "test"].reset_index(drop=True) #predict only test subset
test.rename(columns={"text": "text_original", "generated": "text"}, inplace=True)
test['text'] = [x if str(x) != "nan" else y for (x, y) in zip(test.text, test.text_original)]
test['label'] = ["human" if "human" in x else "machine" for x in test.multi_label]

gc.collect()
torch.cuda.empty_cache()

start = time.time()
with torch.no_grad():
  preds = predict(test)
test['predictions'] = preds[0]
test['prediction_probs'] = preds[1]
end = time.time()
print(f"{output_model.split('/')[-1]} testing took {(end - start)/60} min")
try:
  print(f"{output_model.split('/')[-1]} memory footprint {classifier.model.get_memory_footprint()/1024/1024/1024} GB")
except:
  pass
try:
  report_gpu()
except:
  pass

if "roberta-large-openai-detector" in output_model:
  test['predictions'] = test['predictions'].str.replace('LABEL_1', 'human').str.replace('LABEL_0', 'machine')
if "roberta-base-openai-detector" in output_model:
  test['predictions'] = test['predictions'].str.replace('Real', 'human').str.replace('Fake', 'machine')
if "chatgpt-detector-roberta" in output_model:
  test['predictions'] = test['predictions'].str.replace('Human', 'human').str.replace('ChatGPT', 'machine')
if "roberta-base-autextification-detection" in output_model:
  test['predictions'] = test['predictions'].str.replace('generated', 'machine')
if "ruroberta-ruatd-binary" in output_model:
  test['predictions'] = test['predictions'].str.replace('LABEL_0', 'human').str.replace('LABEL_1', 'machine')

test.to_csv(f"{DATAPATH}{obfuscated_dataset.split('.csv')[0]}_{output_model.split('/')[-1]}.csv.gz", compression='gzip', index=False)
print(f"{DATAPATH}{obfuscated_dataset.split('.csv')[0]}_{output_model.split('/')[-1]}.csv.gz")
print(classification_report(test['label'], test['predictions'], digits=4))
