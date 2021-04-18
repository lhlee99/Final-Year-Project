#%% INIT
import json
import random
import os
import numpy as np
import re
import time
import os
from zipfile import ZipFile
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertModel, RobertaModel, RobertaTokenizerFast, BertTokenizer

#%%
def generate_class_vector(df,
                        model_name="bert-base-chinese",
                        batch_size=None,
                        output_name='output/rename.pt',
                        save=True,
                        roberta=False,
                        # full=False,
                        # full_interval=2
                        ):
    """Generate class vectors with BERT and save the tensor in output."""
    # zip full output
    # if full:
    #     if os.path.exists("tmp"):
    #         os.system('cmd /k "rm -R tmp"')
    #     os.mkdir('tmp')
    #     zipObj = ZipFile(output_name.replace('.pt', f'_full.zip'), 'w')

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

    if roberta:
        model = RobertaModel.from_pretrained(model_name)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except TypeError as e:
            print(e)
            tokenizer = BertTokenizer.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)

    model.to(device)

    concat_output = None
    hidden = None
    data_length = len(df)
    max_length = 512 if not roberta else 256
    batches = np.arange(0, data_length, batch_size)
    log_interval = int(len(batches)/20) if int(len(batches)/20) != 0 else len(batches)//2
    # print(f'max length sequence: {max_length}')
    print(f"number of batches: {len(batches)}")

    start = time.time()
    count = 0
    for batch in tqdm(batches):
        if batch + batch_size >= data_length:
            tokens_tensor = tokenizer(df[batch:], return_tensors="pt", padding='max_length', truncation=True, max_length=max_length) # pt for pytorch
        else:
            tokens_tensor = tokenizer(df[batch:batch + batch_size], return_tensors="pt", padding='max_length', truncation=True, max_length=max_length) # pt for pytorch
        tokens_tensor.to(device)

        with torch.no_grad():
            output = model(**tokens_tensor)

        if concat_output == None:
            concat_output = output.pooler_output
        else:
            concat_output = torch.cat([concat_output, output.pooler_output], 0)

        # if full:
        #     if hidden == None:
        #         hidden = output.last_hidden_state
        #     else:
        #         hidden = torch.cat([hidden, output.last_hidden_state], 0)

        #     if i % (log_interval * full_interval) == 0 and i != 0:
        #         tmp_name = output_name.replace('.pt', f'_full_{count}.pt')
        #         tmp_name = tmp_name.replace('output/', 'tmp/')
        #         torch.save(concat_output, tmp_name)
        #         zipObj.write(tmp_name)
        #         os.remove(tmp_name)
        #         print(f"hidden: {hidden.shape}, saved at {tmp_name}")
        #         count += 1
        #         del hidden
        #         hidden = None

    print(f"Pooler Output Shape: {concat_output.shape}")
    if save:
        torch.save(concat_output, output_name)
        print(f"Written to {output_name}")
    return concat_output
    # if hidden != None:
    #     tmp_name = output_name.replace('.pt', f'_full_{count}.pt')
    #     torch.save(concat_output, tmp_name)
    #     zipObj.write(tmp_name)
    #     os.remove(tmp_name)
    #     print(f"hidden: {hidden.shape}, saved at {tmp_name}")
    # if full:
    #     os.rmdir("tmp")
    #     zipObj.close()

#%% split text function
def split_text(text, max_length, recursive_until=None, step=10):
    """Split text with multiple characters according to max length.
    buffer is accumulated with different segments and it will be splitted if it exceed the max length.
    Return a list of string.
    """
    if len(text) <= max_length:
        return [text]
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
            if recursive_until:
                if max_length+step < recursive_until:
                    return split_text(text, max_length+step, recursive_until=recursive_until)
                else:
                    raise Exception(f'splitted segment is larger than recursive limit {recursive_until}\n{segment}\n{text}')
            else:
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
