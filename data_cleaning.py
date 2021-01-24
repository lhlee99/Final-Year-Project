#%% INIT
import json
import pandas as pd
import random
import os

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

for filename in filenames:
    with open(filename, 'r') as f:
        data = json.loads(f.read())
        for post in data:
            # Assume all posts length is less than 512
            posts.append(post['postName'])

            # Post
            post_content.append(post['postContent'])

            # Comments
            for _comment in post['comments']:
                comment = _comment['content']
                # empty comments
                if comment == '':
                    continue
                # clean comments
                else:
                    comments.append(comment)

# Remove duplicate post and post content
posts = list(dict.fromkeys(posts))
post_content = list(dict.fromkeys(post_content))

df = posts + post_content + comments

print('-'*10)
print(f"Total: {len(df)}")
print(f'number of posts: {len(posts)}')
print(f'number of post content: {len(post_content)}')
print(f'number of comments: {len(comments)}')
print('-'*10)

#%%
with open('vocab.txt', 'r') as f:
    vocab = f.read().split('\n')
vocab = vocab[670:7991]
vocab += [' ', '\n',
    '-', '：', ':', ',', '，', '﹐', '。', '.', 'ㄧ', '？', '?', '！', '!', '；', ';',
    '1', '2', '3','4', '5', '6', '7', '8', '9', '0'
]

#%%
import re
def handle_space_and_newline(string):
    string = re.sub('\n', ' ', string)
    string = re.sub('  +', ' ', string)
    string = string.strip()
    return string

# %%
removed = []
df_clean = []
original = []
for i, content in enumerate(df):
    url_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    filtered = re.sub(url_regex, '', content)
    filtered = ''.join(c for c in filtered if c in vocab)
    filtered = handle_space_and_newline(filtered)
    if len(filtered) > 5:
        df_clean.append(filtered)
        original.append(content)
    else:
        removed.append(content)
    if i % 1000 == 0:
        print(i)
print(len(df_clean))

#%%
df_clean_2 = []
original_2 = []
for content, origin in zip(df_clean, original):
    if len(content) > 512:
        splitted = split_text(content, 512)
        df_clean_2 += splitted
        for i in range(len(splitted)):
            original_2.append(origin)
    else:
        original_2.append(origin)
        df_clean_2.append(content)
print(len(max(df_clean_2, key=len)))
print(max(df_clean_2, key=len))

#%%
filename = 'cleaned_1202_v5'
with open(f"data/{filename}.json", 'w', encoding='utf-8') as f:
    f.write(json.dumps(df_clean_2, ensure_ascii=False))
with open(f"data/{filename}_original.json", 'w', encoding='utf-8') as f:
    f.write(json.dumps(original, ensure_ascii=False))
with open(f"data/{filename}_removed.json", 'w', encoding='utf-8') as f:
    f.write(json.dumps(removed, ensure_ascii=False))
