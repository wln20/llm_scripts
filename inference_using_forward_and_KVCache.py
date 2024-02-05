"""
Description: Use the forward method to do inference with causal llms, along with KV Cache to speed up decoding.
Usage: python inference_using_forward_and_KVCache.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# model
model = AutoModelForCausalLM.from_pretrained("facebook/opt-2.7b", device_map='auto')
# tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")

max_gen_len = 10
in_text = "Four score and seven years ago, our" # len = 9
in_text_orig = in_text

in_tokens = tokenizer(in_text, max_length=50, truncation=True, return_tensors="pt")['input_ids'].to(model.device)

# inference
i = 0
kvcache = None
with torch.no_grad():
    while i < max_gen_len:
        # pass the past_key_values parameter
        output = model(in_tokens, past_key_values=kvcache)
        logits = output.logits
        kvcache = output.past_key_values
        out_token = torch.argmax(logits[-1, :], dim=1, keepdim=True)[-1]  
        in_text += tokenizer.decode(out_token)
        # let the in_tokens for the next round just be this round's out_token
        in_tokens = torch.unsqueeze(out_token, 0)
        print(f'step {i} output: {in_text}', flush=True)
        i += 1


print(f' Input: {in_text_orig}')
print(f'Output: {in_text}')
