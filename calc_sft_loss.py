import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# model
model = AutoModelForCausalLM.from_pretrained("facebook/opt-2.7b", device_map='auto')
# tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")

question = "What is America's capital city?"
answer = "Washington."
# first, only collect the input_ids for the texts and labels
inputs_context = tokenizer(text=question, return_tensors="pt")['input_ids']
inputs_answer = tokenizer(text=answer, add_special_tokens=False, return_tensors="pt")['input_ids']
labels_context = torch.tensor([[tokenizer.pad_token_id for _ in range(inputs_context.shape[1])]])   # <pad> for context's labels
labels_answer = inputs_answer

# concatenate the context and answer to construct inputs
inputs = {}
inputs['input_ids'] = torch.cat([inputs_context, inputs_answer], dim=1).cuda()
inputs['labels'] = torch.cat([labels_context, labels_answer], dim=1).cuda()
inputs['attention_mask'] = torch.tensor([[1 for _ in range(inputs['input_ids'].shape[1])]]).cuda()

print(inputs['input_ids'].shape, inputs['labels'].shape, inputs['attention_mask'].shape)
# make sure the dimension is right
assert inputs['input_ids'].shape == inputs['labels'].shape == inputs['attention_mask'].shape

# turn the attention_mask to 0 for the paddings in input_ids (if exist)
inputs['attention_mask'][0][inputs['input_ids'][0] == tokenizer.pad_token_id] = 0

# turn the paddings in the labels to -100
inputs['labels'][0][inputs['labels'][0] == tokenizer.pad_token_id] = -100

print(inputs)
"""
{'input_ids': tensor([[    2,  2264,    16,   730,    18,   812,   343,   116, 22247,     4]],
       device='cuda:0'), 'labels': tensor([[ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100, 22247,     4]],
       device='cuda:0'), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], device='cuda:0')}
"""

output = model(**inputs)
print(output['loss'])
"""
tensor(6.9341, device='cuda:0', grad_fn=<NllLossBackward0>)
"""
