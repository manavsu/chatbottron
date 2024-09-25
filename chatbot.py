from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_NAME = "google-t5/t5-large"
# MODEL_NAME = "facebook/blenderbot-400M-distill"
CUSTOM_CACHE_DIR = "cache"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=CUSTOM_CACHE_DIR).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CUSTOM_CACHE_DIR, clean_up_tokenization_spaces=True)

conversation_history = []

def chat(input_text):
    history_string = "\n".join(conversation_history)

    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    outputs = model.generate(**inputs, max_length=100)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    conversation_history.append(input_text)
    conversation_history.append(response)

    return response

while (True):
    user_input = input("> ")
    print(chat(user_input))