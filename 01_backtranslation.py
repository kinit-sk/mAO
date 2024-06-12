import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, BitsAndBytesConfig, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import pandas as pd
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use4bit = True
use8bit = False
quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=False, bnb_4bit_compute_type=torch.float16, load_in_4bit=use4bit, load_in_8bit=use8bit, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, int8_threshold=0, int4_threshold=0)

#model_name = "facebook/m2m100-12B-last-ckpt"
model_name = "facebook/m2m100_1.2B"
#model_name = "facebook/m2m100_418M"
#model_name = "facebook/nllb-200-distilled-600M"
model = M2M100ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
#model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16)
#model = M2M100ForConditionalGeneration.from_pretrained(model_name, device_map='auto', quantization_config=quantization_config, offload_state_dict=True, max_memory={0: "40GIB", "cpu": "50GIB"}, offload_folder='./', load_in_4bit=use4bit, load_in_8bit=use8bit, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.to(device)
model = model.half()

def nllb_backtranslate(text, lang):
  via_lang = "en" if lang != "en" else "es"
  
  translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=lang, tgt_lang=via_lang, max_length = 1024)#, device=device)
  output = translator(text)
  translated_text = output[0]['translation_text']
  
  translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=via_lang, tgt_lang=lang, max_length = 1024)#, device=device)
  output = translator(translated_text)
  translated_text = output[0]['translation_text']
  
  return translated_text

def backtranslate(text, lang):
  via_lang = "en" if lang != "en" else "es"
  
  tokenizer.src_lang = lang
  encoded = tokenizer(text, return_tensors="pt").to(device)
  generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(via_lang))
  generated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

  tokenizer.src_lang = via_lang
  encoded = tokenizer(generated, return_tensors="pt").to(device)
  generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(lang))
  generated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

  return generated

test = pd.read_csv('dataset/multitude.csv.gz')
subset = test#[:100]

generated = [""] * len(subset)
model = model.eval()

with torch.no_grad():
  for index, row in tqdm(subset.iterrows(), total=subset.shape[0]):
    if ("generated" in row.index) and (row['generated'] is not np.NaN) and (str(row['generated']) != "nan"):
      generated[index] = row['generated']
      continue
    generated[index] = backtranslate(row.text, row.language)
    #generated[index] = nllb_backtranslate(row.text, row.language)
    subset['generated'] = generated
    if (index % 10) == 0:
      subset.to_csv(f'temp.csv', index=False)

print(pd.DataFrame([subset['text'] == subset['generated']]).T.value_counts())

subset.to_csv(f"dataset/multitude_obfuscated_backtranslated-{model_name.split('/')[-1].replace('-last-ckpt','').replace('_', '-')}.csv.gz", index=False)
