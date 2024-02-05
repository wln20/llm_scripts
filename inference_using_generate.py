"""
Description: Use the generate method to do inference with causal llms.
Usage: python inference_using_generate.py
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
# model
model = AutoModelForCausalLM.from_pretrained("facebook/opt-2.7b", device_map='auto')
# tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")
in_text = "Four score and seven years ago, our"

in_tokens = tokenizer(in_text, return_tensors="pt")['input_ids'].to(model.device)

out = model.generate(in_tokens, max_new_tokens=50)
"""
out.shape = [1,59] 
"""
out_2 = model.generate(in_tokens, output_scores=True, max_new_tokens=50)
"""
out_2.shape = [1,59]
"""
out_3 = model.generate(in_tokens, output_scores=True, return_dict_in_generate=True, max_new_tokens=50)
"""
type(out_3) = <class 'transformers.generation.utils.GreedySearchDecoderOnlyOutput'>
out_3.keys() = odict_keys(['sequences', 'scores'])
out_3.sequences.shape = [1,59]	
type(out_3.scores) = 'tuple'
len(out_3.scores) = 50	
out_3.scores[0].shape = torch.Size([1, 50272])   
"""
output = tokenizer.batch_decode(out)
print(output)
"""
['</s>Four score and seven years ago, our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal.\n\nNow we are engaged in a great civil war, testing whether that nation, or any nation so conceived and']
"""
