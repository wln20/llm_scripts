"""
Description: Download hf models.
Usage: python download_model.py --model_name [MODEL-NAME] --use_auth_token [AUTH-TOKEN]
"""
import os
# change the endpoint for file download, especially useful when it's hard to reach `huggingface.co`
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, help='model name')
parser.add_argument('--use_auth_token', type=str, default=None)
args = parser.parse_args()

if __name__ == "__main__":
    # download tokenizer
    while True:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_auth_token=args.use_auth_token)
        except:
            continue
        break

    # download model
    while True:
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model_name, use_auth_token=args.use_auth_token, device_map="auto", resume_download=True)
        except:
            continue
        break
