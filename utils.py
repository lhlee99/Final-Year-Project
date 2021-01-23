#%% INIT
import json
import random
import os
import numpy as np
import re
import time
import os
from zipfile import ZipFile

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertModel

#%% SAMPLING
"""
sample_idx = random.sample(np.arange(0, len(df)).tolist(), 3000)
df = [df[idx] for idx in sample_idx]
positive_label = [positive_label[idx] for idx in sample_idx]

with open("output/labels.json", 'w') as f:
	f.write(json.dumps(positive_label))

print('-'*10)
print("Sample: ")
print(f'number of comments + postname: {len(df)}')
print(f'positive labels: {len(positive_label)}')
print('-'*10)
"""


#%% TESTING
"""
tokens_tensor = tokenizer(df[0:16], return_tensors="pt", padding=True) # pt for pytorch
with torch.no_grad():
	output = model(**tokens_tensor)
output.pooler_output.shape
# output.hidden_states[0][:,0,:].shape
"""
#%%
def generate_class_vector(df, model_name="bert-base-chinese", batch_size=None, output_name='output/rename.pt', full=False, full_interval=2):
    """Generate class vectors with BERT and save the tensor in output."""
    # zip full output
    if os.path.exists("tmp"):
        os.system('cmd /k "rm -R tmp"')
    os.mkdir('tmp')
    zipObj = ZipFile(output_name.replace('.pt', f'_full.zip'), 'w')

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        if batch_size == None:
            batch_size = 128
    else:
        device = torch.device('cpu')
        if batch_size == None:
            batch_size = 16
    print("-"*50 + "\n|")
    print(f'| Generating tensors\n|\tmodel = {model_name}\n|\tbatch_size = {batch_size}\n|\toutput = {output_name}\n|\tdevice = {device}')
    print("|\n" + "-"*50)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    model.to(device)

    concat_output = None
    hidden = None
    data_length = len(df)
    max_length = len(max(df, key=len))
    batches = np.arange(0, data_length, batch_size)
    log_interval = int(len(batches)/20) if int(len(batches)/20) != 0 else len(batches)//2
    # print(f'max length sequence: {max_length}')
    print(f"number of batches: {len(batches)}")

    start = time.time()
    count = 0
    for i, batch in enumerate(batches):
        if batch_size + batch_size >= data_length:
            tokens_tensor = tokenizer(df[batch:], return_tensors="pt", padding='max_length', max_length=max_length) # pt for pytorch
        else:
            tokens_tensor = tokenizer(df[batch:batch + batch_size], return_tensors="pt", padding='max_length', max_length=max_length) # pt for pytorch
        tokens_tensor.to(device)

        with torch.no_grad():
            output = model(**tokens_tensor)

        if concat_output == None:
            concat_output = output.pooler_output
        else:
            concat_output = torch.cat([concat_output, output.pooler_output], 0)

        if hidden == None:
            hidden = output.last_hidden_state
        else:
            hidden = torch.cat([hidden, output.last_hidden_state], 0)

        if i % (log_interval * full_interval) == 0 and i != 0:
            tmp_name = output_name.replace('.pt', f'_full_{count}.pt')
            tmp_name = tmp_name.replace('output/', 'tmp/')
            torch.save(concat_output, tmp_name)
            zipObj.write(tmp_name)
            os.remove(tmp_name)
            print(f"hidden: {hidden.shape}, saved at {tmp_name}")
            count += 1
            del hidden
            hidden = None

        if i % log_interval == 0:
            print(f'Batch: {i+1} [{(i+1)*batch_size}/{data_length}] {round(time.time() - start, 2)}s {(i+1)*batch_size*1.0/data_length * 100}%')
    print(f"Pooler Output: {concat_output.shape}")
    torch.save(concat_output, output_name)
    if hidden != None:
        tmp_name = output_name.replace('.pt', f'_full_{count}.pt')
        torch.save(concat_output, tmp_name)
        zipObj.write(tmp_name)
        os.remove(tmp_name)
        print(f"hidden: {hidden.shape}, saved at {tmp_name}")

    os.rmdir("tmp")
    zipObj.close()

#%% split text function
def split_text(text, max_length):
    """Split text with multiple characters according to max length.
    buffer is accumulated with different segments and it will be splitted if it exceed the max length.
    Return a list of string.
    """
    breaks = [i for i in re.finditer(' |\n|\：|\:|\,|\，|\﹐|\。|\ㄧ|\？|\?|\！|\!|\；|\;', text)]
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
        mid = len(breaks)//2
        segments = [text[:breaks[mid-1].end()], text[breaks[mid-1].end():]]

    if segments == []:
        raise Exception(f'something is wrong \n{max_length}\n{text}')

    for segment in segments:
        if len(segment) > max_length:
            raise Exception(f'splitted segment is larger than {max_length}\n{segment}\n{text}')
    return segments

def analyze_tokenization(sequence, tokenizer):
    ## this is to examine the tokenization process, one line of code can do all of these
    print(sequence)

    # tokenize (break sentence into words)
    tokens = tokenizer.tokenize(sequence)
    print(tokens)

    # words to id
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(tokens_ids)

    # add special tokens [CLS] [SEP] 101 102
    tokens_ids = tokenizer.build_inputs_with_special_tokens(tokens_ids)
    print(tokens_ids)

    print(tokenizer.convert_ids_to_tokens(tokens_ids))

# %%
