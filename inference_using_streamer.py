"""
Description: Use the TextStreamer to do inference with causal llms, showing the autoregressive generation process.
Usage: python inference_using_streamer.py
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
# model
model = AutoModelForCausalLM.from_pretrained("facebook/opt-2.7b", device_map='auto')
# tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")
in_text = "Four score and seven years ago, our"

in_tokens = tokenizer(in_text, return_tensors="pt")['input_ids'].to(model.device)

streamer = TextStreamer(tokenizer)
output = model.generate(in_tokens, use_cache=True, streamer=streamer, do_sample=False, top_p=None, temperature=None, max_new_tokens=400)

