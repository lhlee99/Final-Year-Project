#%%
import postgres
#%%
from tqdm import tqdm
import re

from utils import split_text
#%%
comments = postgres.query("""
SELECT cleaned_content FROM comment
WHERE tcreated::date < '2020-08-01'
ORDER BY comment_id
""")

def actual_length(sequence):
    tmp = re.sub(r"[A-Za-z']+", "_", sequence)
    return len(tmp)
comments = [i[0].lower() for i in comments if actual_length(i[0]) >= 5]


# %%
def split_text(text, max_length, recursive_until=None, step=10):
    """Split text with multiple characters according to max length.
    buffer is accumulated with different segments and it will be splitted if it exceed the max length.
    Return a list of string.
    """
    if len(text) <= max_length:
        return [text]
    breaks = [i for i in re.finditer(' |\n|\：|\:|\,|\，|\﹐|\。|\ㄧ|\？|\?|\！|\!|\；|\;|\、|\.', text)]
    segments = []
    start_offset = 0
    for k, p in enumerate(breaks):
        if p.end() - start_offset > max_length:
            start = start_offset
            end = breaks[k-1].end()
            segment = text[start:end]
            start_offset = breaks[k-1].end()
            segments.append(segment)

    if segments == []:
        if len(breaks) == 0:
            if len(text) < max_length:
                return [text]
            else:
                return [text[:recursive_until]]
        else:
            mid = len(breaks)//2
            segments = [text[:breaks[mid-1].end()], text[breaks[mid-1].end():]]

    if segments == []:
        raise Exception(f'something is wrong \n{max_length}\n{text}')

    for segment in segments:
        if len(segment) > max_length:
            if recursive_until:
                if max_length+step < recursive_until:
                    return split_text(text, max_length+step, recursive_until=recursive_until)
                else:
                    return [text[:recursive_until]]
                    # raise Exception(f'splitted segment is larger than recursive limit {recursive_until}\n{segment}\n{text}')
            else:
                raise Exception(f'splitted segment is larger than {max_length}\n{segment}\n{text}')
    return segments
splitted_comments = [split_text(i, 40, recursive_until=128) for i in comments]

# %%
with open("pretraining_dataset.txt", 'w') as f:
    tmp = '\n\n'.join(['\n'.join(i) for i in splitted_comments])
    f.write(tmp + '\n')

#%% CUSTOM TOKEN
tmp_freq = {}
for comment in tqdm(comments, desc="Parsing"):
    eng_vocabs = re.findall(r"[A-Za-z']+", comment)
    eng_vocabs = [i.lower().strip() for i in eng_vocabs]
    for vocab in eng_vocabs:
        if vocab in tmp_freq:
            tmp_freq[vocab] += 1
        else:
            tmp_freq[vocab] = 1
freq = [(v, k) for (k, v) in tmp_freq.items()]
freq.sort(reverse=True)

#%%
new_tokens = [i[1] for i in freq if i[0] >= 100]

# %%
import json
with open('new_token.json', 'w') as f:
    f.write(json.dumps(new_tokens))
# %%
