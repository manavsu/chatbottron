
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, Mamba2ForCausalLM, AutoModelWithLMHead
import torch
import bitsandbytes as bnb

TOKEN = open("token.secret").read().strip()
CUSTOM_CACHE_DIR = "cache"
MODEL_NAME = "google/gemma-2-2b"
CORES = 16

torch.set_num_threads(16)

device = "cuda" if torch.cuda.is_available() else "cpu"


model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=TOKEN, cache_dir=CUSTOM_CACHE_DIR).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=TOKEN, cache_dir=CUSTOM_CACHE_DIR)


prompt = "where is nmap stored on arch?"
inputs =  tokenizer([prompt], return_tensors="pt").to(device)
generated = model.generate(**inputs, max_length=50)
output = tokenizer.decode(generated[0])
print(generated)
print(type(output))
print(output)

prompt = "where is nmap stored on arch?"

inputs =  tokenizer([prompt], return_tensors="pt").to(device)
generated = model.generate(**inputs, max_length=50)
