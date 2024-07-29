import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# model
model = AutoModelForCausalLM.from_pretrained("facebook/opt-2.7b", device_map='auto')
# tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")

in_text = "Four score and seven years ago, our" # len = 9
inputs = tokenizer(text=in_text, text_target=in_text, return_tensors="pt")
for key, value in inputs.items():   # move to cuda
    inputs[key] = value.cuda()
print(inputs)
"""
{'input_ids': tensor([[    2, 22113,  1471,     8,   707,   107,   536,     6,    84]],
       device='cuda:0'), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]], device='cuda:0'), 'labels': tensor([[    2, 22113,  1471,     8,   707,   107,   536,     6,    84]],
"""

output = model(**inputs)
print(output.keys())
"""
odict_keys(['loss', 'logits', 'past_key_values'])
"""
print(output['loss'])
"""
tensor(2.5574, device='cuda:0', grad_fn=<NllLossBackward0>)
"""
