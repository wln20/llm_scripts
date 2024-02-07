"""
Description: Do sentence completion tasks with several models simultaneously, convenient for comparing multiple models' responses to a same prompt.
Usage: First fill in the list `model_paths` with the paths of the models, then run: python sentence_completion.py
"""
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

model_paths = [
  path/to/model1,
  path/to/model2,
  ...
    ]
models = [AutoModelForCausalLM.from_pretrained(model_paths[i], trust_remote_code=True, device_map='auto') for i in range(len(model_paths))]
tokenizers = [AutoTokenizer.from_pretrained(model_paths[i], trust_remote_code=True) for i in range(len(model_paths))]

while True:
    print('=========================================')
    in_text = input('input: ')
    if in_text == 'q':
        break
    for i in range(len(model_paths)):
        in_tokens = tokenizers[i](in_text, return_tensors="pt")['input_ids'].to(models[i].device)
        out = models[i].generate(in_tokens, max_new_tokens=100)
        output = tokenizers[i].batch_decode(out)[0]
        print(f"Model {os.path.split(model_paths[i])[-1]}'s response: \n{output}")
        print('-------------------------------')
