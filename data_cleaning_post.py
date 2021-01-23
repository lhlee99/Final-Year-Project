#%% INIT
import json
import pandas as pd
import random

from transformers import AutoTokenizer

from utils import split_text, generate_class_vector, analyze_tokenization
random.seed(0)

#%% IMPORT DATA
tmp = os.popen("ls data/20201202-*").read()
filenames = tmp.strip().split('\n')
# filenames = [f'data/{i}' for i in filenames]

posts = []
post_content = []
comments = []
positive_label_post = []
positive_label_comment = []

for filename in filenames:
    with open(filename, 'r') as f:
        data = json.loads(f.read())
        for post in data:
            post_content.append(f"{post['postName']}π{post['postContent']}")

# Remove duplicate post and post content
post_content = list(dict.fromkeys(post_content))

df = post_content

print('-'*10)
print(f"Total: {len(df)}")
print('-'*10)

#%%
with open('vocab.txt', 'r') as f:
    vocab = f.read().split('\n')
vocab = vocab[670:7991]
vocab += [' ', '\n',
    '-', '：', ':', ',', '，', '﹐', '。', '.', 'ㄧ', '？', '?', '！', '!', '；', ';',
    '1', '2', '3','4', '5', '6', '7', '8', '9', '0', 'π'
]

#%%
import re
def handle_space_and_newline(string):
    string = re.sub('\n', ' ', string)
    string = re.sub('  +', ' ', string)
    string = string.strip()
    return string

# %%
df_clean = []
for i, content in enumerate(df):
    url_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    filtered = re.sub(url_regex, '', content)
    filtered = ''.join(c for c in filtered if c in vocab)
    filtered = handle_space_and_newline(filtered)
    df_clean.append(filtered)
print(len(df_clean))

#%%
df_clean_2 = []
for content in df_clean:
    if len(content) > 511:
        name, post_content = content.split('π')
        for i in split_text(post_content, 511 - len(name)):
            df_clean_2.append(name + ' ' + i)
    else:
        df_clean_2.append(re.sub('π', ' ', content))
print(len(max(df_clean_2, key=len)))
print(max(df_clean_2, key=len))

#%%
filename = 'post_1202_v1'
with open(f"data/{filename}.json", 'w', encoding='utf-8') as f:
    f.write(json.dumps(df_clean_2, ensure_ascii=False))
# with open(f"data/{filename}_original.json", 'w', encoding='utf-8') as f:
#     f.write(json.dumps(df_clean, ensure_ascii=False))
# with open(f"data/{filename}_removed.json", 'w', encoding='utf-8') as f:
#     f.write(json.dumps(df_clean, ensure_ascii=False))

# %%
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
analyze_tokenization('日, .清', tokenizer)
# %%
tokens_tensor = tokenizer(df_clean, return_tensors="pt", padding=True) # pt for pytorch
