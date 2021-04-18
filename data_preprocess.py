#%%
import numpy as np
import postgres
import matplotlib.pyplot as plt
import numpy as np
import json
import re
from tqdm import tqdm, trange
from sqlalchemy import create_engine
import pandas as pd
import configparser

#%%
try:
    f = open('post_content/all_comment.json')
    comments = json.loads(f.read())
except FileNotFoundError as e:
    print("File not found, requesting from database")
    data = postgres.query("SELECT comment_id, author_id, tcreated, post_id, content, set_id, cluster_id FROM comment ORDER BY comment_id", return_dict=True)
    # with open("post_content/all_comment.json", 'w', encoding='utf-8') as f:
    #     f.write(json.dumps(data, ensure_ascii=False))

#%% CLEANING
# url_regex (unused) = r"[-a-zA-Z0-9@:%_\+.~#?&//=]{2,256}\.[a-z]{2,4}\b(\/[-a-zA-Z0-9@:%_\+.~#?&//=]*)?"
def clean(sequence):
    url_regex = r"(([\w-]+:\/\/?|www[.])[^\s()<>]+(?:\([\w\d]+\)|([^[:punct:]\s]|\/)\S+))"
    img_regex = r"\S+\.(jpg|png|gif) \([\d.]+ KB\) \d{4}-\d{1,2}-\d{1,2} \d{2}:\d{2} P?A?M"
    lihkg_tag_regex = r"\[\/?(url|img|size=\d|b|i|u|s|red|green|blue|purple|violet|brown|pink|orange|gold|maroon|teal|limegreen|left|right|center|quote|member)\]|\[\/.+"
    chinese_regex = r"\[ 本帖最後由.+編輯 \]|引用: 原帖由.+發表| - 分享自 LIHKG 討論區"
    time_regex = r"\d{4}-\d{1,2}-\d{1,2} \d{2}:\d{2} P?A?M"
    all_regex = re.compile('|'.join([url_regex, img_regex, lihkg_tag_regex, chinese_regex, time_regex]))
    return re.sub(all_regex, '', content)


#%%
for d in tqdm(data, desc="Cleaning"):
    content = d['content']
    cleaned = re.sub(all_regex, '', content)
    d['cleaned_content'] = cleaned

#%% drop table and reupload everything as update is too slow
# postgres.update_many('comment', comments, pk='comment_id')
comment_df = pd.DataFrame(data)
#%%
config = configparser.ConfigParser()
config.read('db.ini')
conn = psycopg2.connect( host=config['postgres']['host'],
                        user=config['postgres']['user'],
                        password=config['postgres']['passwd'],
                        dbname=config['postgres']['db'],
                        # charset='utf8mb4',
                        )
engine = create_engine(f'postgresql://postgres:{config['postgres']['passwd']}@{config['postgres']['host']}:5432/{config['postgres']['passwd']}')

# 827231

#%% CUSTOM TOKEN
tmp_freq = {}
for comment in tqdm(comments, desc="Parsing"):
    filtered = re.sub(all_regex, '', comment)
    eng_vocabs = re.findall(r"[A-Za-z']+", filtered)
    eng_vocabs = [i.lower().strip() for i in eng_vocabs]
    for vocab in eng_vocabs:
        if vocab in tmp_freq:
            tmp_freq[vocab] += 1
        else:
            tmp_freq[vocab] = 1

#%%
freq = [(v, k) for (k, v) in tmp_freq.items()]
freq.sort(reverse=True)
with open("custom_vocab.json", 'w', encoding='utf-8') as f:
    f.write(json.dumps(freq, ensure_ascii=False))
# %% Token evaluation
count = 0
for comment in comments:
    filtered = re.sub(url_regex, '', comment)
    eng_vocabs = re.findall(r'[A-Za-z]+', filtered)
    eng_vocabs = [i.lower().strip() for i in eng_vocabs]
    for vocab in eng_vocabs:
        if vocab in tmp:
            tmp[vocab].append((comment, filtered))
            count += 1
    if count > 1000:
        break
# %%
