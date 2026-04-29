import os
import requests
import tiktoken
import numpy as np

input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

n = len(data)
train_data = data[:int(n*0.7)]
val_data = data[int(n*0.7):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
"""
special = {
    "<SOS>": 50257,
    "<PROMPT>": 50258,
    "</PROMPT>": 50259,
    "<TITLE>": 50260,
    "</TITLE>": 50261,
    "<INGREDIENTS>": 50262,
    "</INGREDIENTS>": 50263,
    "<DIRECTIONS>": 50264,
    "</DIRECTIONS>": 50265,
    "<EOS>": 50266,
}
"""

"""
enc = tiktoken.Encoding(
    name="custom-gpt2",
    pat_str=base._pat_str,
    mergeable_ranks=base._mergeable_ranks,
    special_tokens=special,
)
"""

train_ids = enc.encode(train_data, allowed_special='all')
val_ids = enc.encode(val_data, allowed_special='all')
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))


