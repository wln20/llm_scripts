"""
Description: Use the naive forward method to do inference with causal llms.
Usage: python inference_using_forward.py
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
"""
in_tokens.size = [1,9]
"""


# inference
i = 0
with torch.no_grad():
    while i < max_gen_len:
        output = model(in_tokens)
        logits = output.logits      
        """
        **For the first round:**
        logits.size = [1,9,50272]
        """
        out_token = torch.argmax(logits[-1, :], dim=1, keepdim=True)[-1]  
        """
        **For the first round:**
        logits[-1, :].size = [9,52072]
        torch.argmax(logits[-1, :], dim=1, keepdim=True).size = [9,1]
        out_token.size = [1]
        """  
        in_text += tokenizer.decode(out_token)
        in_tokens = torch.unsqueeze(torch.cat((in_tokens[0], out_token)),0)
        """
        **For the first round:**
        in_tokens.size = [1,9]
        in_tokens[0].size = [9]
        out_token.size = [1]
        torch.unsqueeze(torch.cat((in_tokens[0], out_token)),0).size = [1,10]
        """
        print(f'step {i} output: {in_text}', flush=True)
        i += 1


out_text = tokenizer.batch_decode(in_tokens)[0].replace('</s>', '')
"""
tokenizer.batch_decode(in_tokens).size = [1,19] 
"""
print(f' Input: {in_text_orig}')
print(f'Output: {out_text}')
"""
step 0 output: Four score and seven years ago, our fathers
step 1 output: Four score and seven years ago, our fathers brought
step 2 output: Four score and seven years ago, our fathers brought forth
step 3 output: Four score and seven years ago, our fathers brought forth on
step 4 output: Four score and seven years ago, our fathers brought forth on this
step 5 output: Four score and seven years ago, our fathers brought forth on this continent
step 6 output: Four score and seven years ago, our fathers brought forth on this continent,
step 7 output: Four score and seven years ago, our fathers brought forth on this continent, a
step 8 output: Four score and seven years ago, our fathers brought forth on this continent, a new
step 9 output: Four score and seven years ago, our fathers brought forth on this continent, a new nation
 Input: Four score and seven years ago, our
Output: Four score and seven years ago, our fathers brought forth on this continent, a new nation
"""
