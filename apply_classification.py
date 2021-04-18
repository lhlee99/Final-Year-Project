#%% INIT
print("Importing")

from transformers import BertTokenizerFast, BertForSequenceClassification

import json
import os
import argparse

import postgres
import topic_modeling
import torch
from torch.utils.data import DataLoader
import topic_modeling

from tqdm import tqdm
#%% parse arguments
parser = argparse.ArgumentParser(description='Do validation')
parser.add_argument('-i', '--input', type=str, help='dataset location', required=True)
parser.add_argument('-o', '--output', type=str, help='output location', default=None)
args = parser.parse_args()
print(args)
N_CLASS = 8
SAVE = args.output != None
if SAVE:
    try:
        os.makedirs(args.output)
    except:
        print(f"Directory {args.output} exist")


#%%
print("Reading data")
with open(args.input, 'r') as f:
    sequences = json.loads(f.read())

# %%
def get_labels(tokens, model):
    output = model(**tokens)
    with torch.no_grad():
        prob = torch.nn.functional.softmax(output.logits, dim=1)
        for prediction in prob:
            prediction_id = int(torch.argmax(prediction))
            yield prediction_id

model_name = 'models/lihkgBERT-class'

print("Tokenizes dataset")
tokenizer = BertTokenizerFast.from_pretrained(model_name)
tokenizer_params = {
    'truncation': True,
    'padding': 'max_length',
    'return_tensors': 'pt',
    'max_length': 512
}
token_tensors = tokenizer(sequences, **tokenizer_params)
class LIHKGDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
        print(f"Dataset length: {len(self.encodings['input_ids'])}")

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])
dataset = LIHKGDataset(token_tensors)
dataloader = DataLoader(dataset, batch_size=16)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print('Loading model')
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=N_CLASS)
model.to(device)
labels = []

#%%
for batch in tqdm(dataloader):
    for key in batch.keys():
        batch[key] = batch[key].to(device)
    labels += get_labels(batch, model)

translation = 'hk, china, us, taiwan, japan, uk, other, international, canada, australia'.split(', ')
topics = topic_modeling.termFrequency(sequences, labels, k=N_CLASS, output_dir=f"{args.output}", save=SAVE, translation=translation)

#%%
if SAVE:
    with open(f'{args.output}/result.json', 'w') as f:
        f.write(json.dumps({
            'labels': labels,
            'content': sequences
        }))

#%%
def print_topics(topics, distribution):
    translation = 'hk, china, us, taiwan, japan, uk, other, international, canada, australia'.split(', ')
    for key, value in topics.items():
        print(f"{translation[key]}({key}): {distribution[distribution.Topic == key].Size.values[0]}\n{'-'*20}")
        if len(value) == 20:
            idx = 0
            for i in range(4):
                for j in range(5):
                    to_print = f"{value[idx][0]} [{value[idx][1]:.4f}]"
                    idx += 1
                    print(f'{to_print:<20}', end='')
                print('')
        print('')


print(model_name)
print('-'*40)
print_topics(*topics)
