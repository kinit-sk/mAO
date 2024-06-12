DATAPATH = "./dataset/"
MODELPATH = "./models/"
CACHE = "./cache/"
offload_folder = "./offload_folder/"

import sys

PRE_TRAINED_MODEL_NAME = sys.argv[1] #model identifier or path
model_name = PRE_TRAINED_MODEL_NAME.split('/')[-1]
dataset = 'all' #sys.argv[2] #'en', 'es', 'ru', 'all', 'en3'
generative_model = 'all' #sys.argv[3] #'text-davinci-003', 'gpt-3.5-turbo', 'gpt-4', 'llama-65b', 'opt-66b', 'opt-iml-max-1.3b', 'all', 'combination2', 'combination3'
output_model = f'{MODELPATH}{model_name}-finetuned-{dataset}-{generative_model}'

balance = True
if balance:
    output_model = f'{MODELPATH}{model_name}-finetuned-{dataset}-{generative_model}-balanced'
obfuscated = True #if False original model will be trained (without adversarial samples)
if obfuscated:
    obfuscator = sys.argv[2] #'all' 'dftfooler' 'gptzzzs' 'gptzerobypasser' 'HomoglyphAttack' 'dipper' 'pegasus-paraphrase' 'paraphrased-ChatGPT' 'backtranslated-m2m100-1.2B' 'backtranslated-nllb-200-distilled-1.3B' 'alison'
    output_model = f'{MODELPATH}{model_name}-finetuned-{dataset}-{generative_model}.{obfuscator}'

import os
os.environ['HF_HOME'] = CACHE

import pandas as pd
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from transformers.optimization import Adafactor, AdafactorSchedule
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
import torch
import gc
import nvidia_smi, psutil, shutil
import time

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

def report_gpu():
  nvidia_smi.nvmlInit()
  handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
  info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
  print("GPU [GB]:", f'{info.used/1024/1024/1024:.2f}', "/", f'{info.total/1024/1024/1024:.1f}')
  nvidia_smi.nvmlShutdown()
  print('RAM [GB]:', f'{psutil.virtual_memory()[3]/1024/1024/1024:.2f}', "/", f'{psutil.virtual_memory()[0]/1024/1024/1024:.1f}')

label_names = ["human", "machine"] #0, 1
id2label = {idx:label for idx, label in enumerate(label_names)}
label2id = {v:k for k,v in id2label.items()}

def map_labels(example):
  label_name = example["label"]
  return {"label": label2id[label_name], "label_name": label_name}
    
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, cache_dir=CACHE)

if tokenizer.pad_token is None:
  if tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token
  else:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

start = time.time()
num_labels = len(label_names)
model = AutoModelForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, cache_dir=CACHE, num_labels=num_labels, label2id=label2id, id2label=id2label, ignore_mismatched_sizes=True)
model.resize_token_embeddings(len(tokenizer))
try:
  model.config.pad_token_id = tokenizer.get_vocab()[tokenizer.pad_token]
except:
  print("Warning: Exception occured while setting pad_token_id")
end = time.time()
print(f'{model_name} loading took {(end - start)/60} min')
print(f'{model_name} memory footprint {model.get_memory_footprint()/1024/1024/1024}')
report_gpu()

train = pd.read_csv(DATAPATH + 'multitude_obfuscated_original.csv.gz')
train = train[train.split == "train"]

#language selection
if dataset == "en":
    train = train[train.language == "en"].groupby(['multi_label']).apply(lambda x: x.sample(min(1000, len(x)), random_state = RANDOM_SEED)).sample(frac=1., random_state = 0).reset_index(drop=True)
elif dataset == "en3":
    train = train[train.language == "en"].sample(frac=1., random_state = RANDOM_SEED).reset_index(drop=True)
elif dataset == "es":
    train = train[train.language == "es"].sample(frac=1., random_state = RANDOM_SEED).reset_index(drop=True)
elif dataset == "ru":
    train = train[train.language == "ru"].sample(frac=1., random_state = RANDOM_SEED).reset_index(drop=True)
elif dataset == "all":
    train_en = train[train.language == "en"].groupby(['multi_label']).apply(lambda x: x.sample(min(1000, len(x)), random_state = RANDOM_SEED)).sample(frac=1., random_state = 0).reset_index(drop=True)
    train_es = train[train.language == "es"]
    train_ru = train[train.language == "ru"]
    train = pd.concat([train_en, train_es, train_ru], ignore_index=True, copy=False).sample(frac=1., random_state = RANDOM_SEED).reset_index(drop=True)

#machine-text generation model selection
if generative_model == "combination2":
    train = train[train.multi_label.isin(['human', 'gpt-4', 'vicuna-13b'])].sample(frac=1., random_state = RANDOM_SEED).reset_index(drop=True)
elif generative_model == "combination2a":
    train = train[train.multi_label.isin(['human', 'gpt-4', 'opt-66b'])].sample(frac=1., random_state = RANDOM_SEED).reset_index(drop=True)
elif generative_model == "combination2b":
    train = train[train.multi_label.isin(['human', 'vicuna-13b', 'opt-66b'])].sample(frac=1., random_state = RANDOM_SEED).reset_index(drop=True)
elif generative_model == "combination3":
    train = train[train.multi_label.isin(['human', 'gpt-4', 'vicuna-13b', 'opt-66b'])].sample(frac=1., random_state = RANDOM_SEED).reset_index(drop=True)
elif generative_model != "all":
    train = train[train.multi_label.str.contains("human") | train.multi_label.str.contains(generative_model)].sample(frac=1., random_state = RANDOM_SEED).reset_index(drop=True)

#obfuscator data selection
#obfuscators = ['backtranslated-m2m100-1.2B', 'backtranslated-nllb-200-distilled-1.3B', 'pegasus-paraphrase', 'dipper', 'paraphrased-ChatGPT', 'gptzzzs', 'gptzerobypasser', 'HomoglyphAttack', 'alison', 'dftfooler']
obfuscators = ['backtranslated-m2m100-1.2B', 'backtranslated-nllb-200-distilled-1.3B', 'paraphrased-ChatGPT', 'gptzzzs', 'gptzerobypasser', 'HomoglyphAttack', 'alison', 'dftfooler']
train['obfuscator'] = 'original'
if obfuscated:
  if obfuscator != 'all':
    obfuscators = [obfuscator]
  for o in obfuscators:
    train_obfuscated = pd.read_csv(DATAPATH + f'multitude_obfuscated_{o}.csv.gz')
    train_obfuscated = train_obfuscated[train_obfuscated.split == "train"]
    train_obfuscated = train_obfuscated[~train_obfuscated.multi_label.str.contains("human")]
    train_obfuscated['text'] = train_obfuscated['generated']
    train_en = train_obfuscated[train_obfuscated.language == "en"].groupby(['multi_label']).apply(lambda x: x.sample(min(1000, len(x)), random_state = RANDOM_SEED)).sample(frac=1., random_state = 0).reset_index(drop=True)
    train_es = train_obfuscated[train_obfuscated.language == "es"]
    train_ru = train_obfuscated[train_obfuscated.language == "ru"]
    train_obfuscated = pd.concat([train_en, train_es, train_ru], ignore_index=True, copy=False)#.sample(frac=1/len(obfuscators), random_state = RANDOM_SEED).reset_index(drop=True) #subsample obfuscated data in order to unobfuscated contained at least 50%
    train_obfuscated['obfuscator'] = o
    train = pd.concat([train, train_obfuscated], ignore_index=True, copy=False).sample(frac=1., random_state = RANDOM_SEED).reset_index(drop=True)
    
train['label'] = ["human" if "human" in x else "machine" for x in train.multi_label]

train = train.drop_duplicates(subset=['text'], ignore_index=True)

valid = train[-(len(train)//10):]
train = train[:-(len(train)//10)]

#rebalance 50% original 50% obfuscated with even portion for each obfuscator
if not obfuscated: obfuscators = ['original']
train = pd.concat([train[train.obfuscator == 'original'], train[train.obfuscator != 'original'].groupby(['obfuscator']).apply(lambda x: x.sample(len(train[train.obfuscator == 'original']) // len(obfuscators), replace=True, random_state = RANDOM_SEED))], ignore_index=True, copy=False).sample(frac=1., random_state = RANDOM_SEED).reset_index(drop=True)
valid = pd.concat([valid[valid.obfuscator == 'original'], valid[valid.obfuscator != 'original'].groupby(['obfuscator']).apply(lambda x: x.sample(len(valid[valid.obfuscator == 'original']) // len(obfuscators), replace=True, random_state = RANDOM_SEED))], ignore_index=True, copy=False).sample(frac=1., random_state = RANDOM_SEED).reset_index(drop=True)

if(balance): #balance human vs machine by upsampling minority class
  train = train.groupby(['label']).apply(lambda x: x.sample(train.label.value_counts().max(), replace=True, random_state = RANDOM_SEED)).sample(frac=1., random_state = RANDOM_SEED).reset_index(drop=True)
  valid = valid.groupby(['label']).apply(lambda x: x.sample(valid.label.value_counts().max(), replace=True, random_state = RANDOM_SEED)).sample(frac=1., random_state = RANDOM_SEED).reset_index(drop=True)

print(train.groupby('language')['multi_label'].value_counts())
print(train.label.value_counts())
print(train.obfuscator.value_counts())

train = Dataset.from_pandas(train, split='train')
valid = Dataset.from_pandas(valid, split='validation')
train = train.map(map_labels)
valid = valid.map(map_labels)

def tokenize_texts(examples):
  return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_train = train.map(tokenize_texts, batched=True)
tokenized_valid = valid.map(tokenize_texts, batched=True)

batch_size = 16
gradient_accumulation_steps=4
num_train_epochs = 10
learning_rate=2e-4
metric_for_best_model = 'MacroF1'
logging_steps = len(tokenized_train) // (batch_size * num_train_epochs)
logging_steps = round(2000 / (batch_size * gradient_accumulation_steps)) #eval around each 2000 samples

if '.noearlystopping' in output_model:
  num_train_epochs = 2

if ("small" in model_name):
    #logging_steps //= 3
    logging_steps *= 5
    #learning_rate=2e-8
    metric_for_best_model = 'ACC'

use_fp16 = True
if "mdeberta" in model_name: use_fp16 = False

args = TrainingArguments(
    output_dir=output_model,
    evaluation_strategy = "steps",
    logging_steps = logging_steps, #50,
    save_strategy="steps",
    save_steps = logging_steps, #50,
    save_total_limit=10,
    load_best_model_at_end=True,
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=True,
    num_train_epochs=num_train_epochs,
    weight_decay=0.01,
    push_to_hub=False,
    report_to="none",
    metric_for_best_model = metric_for_best_model,
    fp16=use_fp16 #mdeberta not working with fp16
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"ACC": accuracy_score(labels, predictions), "MacroF1": f1_score(labels, predictions, average='macro'), "MAE": mean_absolute_error(labels, predictions)}

optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
class MyAdafactorSchedule(AdafactorSchedule):
    def get_lr(self):
        opt = self.optimizer
        if "step" in opt.state[opt.param_groups[0]["params"][0]]:
            #lrs = [opt._get_lr(group, opt.state[p]).item() for group in opt.param_groups for p in group["params"]]
            lrs = [opt._get_lr(group, opt.state[p]) for group in opt.param_groups for p in group["params"]]
        else:
            lrs = [args.learning_rate] #just to prevent error in some models (mdeberta), return fixed value according to set TrainingArguments
        return lrs #[lrs]
lr_scheduler = MyAdafactorSchedule(optimizer)#AdafactorSchedule(optimizer)
#optimizers=(optimizer, lr_scheduler)
trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=10)],
    optimizers=(optimizer, lr_scheduler)
)
if '.noearlystopping' in output_model:
  trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, lr_scheduler)
  ) 

start = time.time()
trainer.train()
end = time.time()
print(f'{model_name} memory footprint {model.get_memory_footprint()/1024/1024/1024}')
print(f'{model_name} fine-tuning took {(end - start)/60} min')
report_gpu()

start = time.time()
shutil.rmtree(output_model, ignore_errors=True)
trainer.save_model()
end = time.time()
print(f'{output_model} saving took {(end - start)/60} min')