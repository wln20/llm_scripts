"""
Description: Calculate the number of tokens in a dataset
Usage: python calc_token_num.py --ds_name_or_path [DS-NAME-OR-PATH] --features [FEATURES] [--is_hf_ds] --tokenizer_path [TOKENIZER-PATH]
"""
import argparse
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--ds_name_or_path', type=str, help='path to a local dataset or a hf dataset's name')
parser.add_argument('--features', type=str, default='all', help='the column of the datasets to be considered in the calculation, default to all the columns')
parser.add_argument('--is_hf_ds', type=bool, action='store_true', help='whether this is a local dataset of a hugginggface dataset')
parser.add_argument('--tokenizer_path', type=str, help='path to the tokenizer directory')
args = parser.parse_args()

dataset_name_or_path = args.ds_name_or_path

def calc_num(ds, tokenizer, features='all'):
    features_ls = [feature for feature in ds.features] if features == 'all' else features.split(',')
    print(f"Calculated features: {features_ls}")
    token_num = 0
    item_num = 0
    for feature in features_ls:
        for sentence in tqdm(ds[feature]):
            token_num_this = len(tokenizer(sentence)['input_ids'])
            # ignore the start token
            if 'glm' in tokenizer.name_or_path.lower():  # for Chinese tokenization
                token_num_this -= 2 
            else:
                token_num_this -= 1
            token_num += token_num_this
            item_num += 1
    return token_num, item_num

def calc_num_wrapper(ds, tokenizer, features='all'):
    token_num, item_num = calc_num(ds, tokenizer, features)
    if token_num < 1e3:
        unit_flag = 0
    elif token_num < 1e6:
        unit_flag = 1    # K
    elif token_num < 1e9:
        unit_flag = 2    # M
    elif token_num < 1e12:
        unit_flag = 3    # G
    elif token_num < 1e15:
        unit_flag = 4    # T
    else:
        unit_flag = 5    # P

    unit_map = {
       0 : '',
       1 : 'K', # 1e3
       2 : 'M', # 1e6
       3 : 'G', # 1e9
       4 : 'T', # 1e12
       5 : 'P'  # 1e15
    }
    print('=========================================')
    print(f"Total number of tokens: {token_num / eval(f'1e{unit_flag*3}'):.2f} {unit_map[unit_flag]}")
    print(f"Average sentence length: {token_num / item_num:.2f}")
    print('=========================================')

if __name__ == '__main__':
    ds = load_dataset(dataset_name_or_path) if args.is_hf_ds else load_from_disk(dataset_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)  
    calc_num_wrapper(ds, tokenizer, args.features)
    
