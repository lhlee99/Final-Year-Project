#%%
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AdamW

from tqdm import tqdm
import re
import json
import os

import postgres

#%%
output_dir = "models/lihkgBERT-class"

def mkdir(directory):
    try:
        os.mkdir(directory)
        print(f'Directory {directory} created.')
    except:
        print(f'Directory {directory} already exist.')
        pass

mkdir(output_dir)

# %%
model_name = 'models/lihkgBERT'
print(f"model: {model_name}")
print("Setting up tokenizer")
tokenizer = BertTokenizerFast.from_pretrained(model_name)
print("Setting up model")
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=8)


#%% READ DATASET
print("Loading dataset")
data = postgres.query("""
SELECT title, class_id
FROM post
WHERE class_id IS NOT null
""", return_dict=True)
texts = [i['title'] for i in data]
labels = [i['class_id'] for i in data]

def group_8_n_9(label):
    if label in [8, 9]:
        return 7
    else:
        return label
labels = [group_8_n_9(i) for i in labels]


#%%
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=.3, random_state=42)
test_texts, val_texts, test_labels, val_labels = train_test_split(test_texts, test_labels, test_size=.66, random_state=42)

with open(f'{output_dir}/dataset.json', 'w') as f:
    dataset = {}
    dataset['train'] = {'texts': train_texts, 'labels': train_labels}
    dataset['test'] = {'texts': test_texts, 'labels': test_labels}
    dataset['val'] = {'texts': val_texts, 'labels': val_labels}
    f.write(json.dumps(dataset))

print("Tokenizing dataset")
params = {
    'truncation': True,
    'padding': 'max_length',
    'return_tensors': 'pt',
    'max_length': 256
}
train_encodings = tokenizer(train_texts, **params)
val_encodings = tokenizer(val_texts, **params)
test_encodings = tokenizer(test_texts, **params)

class LIHKGDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        print(f"Dataset length: {len(self.labels)}")

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = LIHKGDataset(train_encodings, train_labels)
val_dataset = LIHKGDataset(val_encodings, val_labels)
test_dataset = LIHKGDataset(test_encodings, test_labels)

# %%
# training_args = TrainingArguments(
#     output_dir=f"./{output_dir}",
#     overwrite_output_dir=True,
#     per_device_train_batch_size=8,
#     # per_device_eval_batch_size=64,
#     num_train_epochs=5,
#     save_strategy='epoch',
#     warmup_steps=0,
#     seed=42,
#     logging_dir=f'./{output_dir}/logs',
#     logging_steps=10,
# )
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
# )
# print("Start Training")
# trainer.train()

# print("Finish training, save model")
# trainer.save_model(f"./{output_dir}")
# tokenizer.save_pretrained(f"./{output_dir}")

#%%
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

optim = AdamW(model.parameters(), lr=5e-5)

def train(loss_vector, accuracy_vector, val_loss_vector, val_accuracy_vector, epoch):
    model.train()
    log_interval = 20
    for i, batch in enumerate(train_loader):
        correct = 0
        x = epoch + i/len(train_loader)
        optim.zero_grad()
        # input_ids = batch['input_ids'].to(device)
        # attention_mask = batch['attention_mask'].to(device)
        # labels = batch['labels'].to(device)
        for key in batch.keys():
            batch[key] = batch[key].to(device)

        output = model(**batch)

        loss = output.loss
        loss.backward()
        optim.step()
        with torch.no_grad():
            pred = torch.argmax(output.logits, dim=1)
            correct = pred.eq(batch['labels']).sum()
            loss_vector.append((x, output.loss.item()))
            accuracy = 100. * correct.to(torch.float32) / len(batch['labels'])
            accuracy_vector.append((x, accuracy.item()))
        if i % log_interval == 0 or i + 1 == len(train_loader):
            print(f'[epoch {x:.2f}] training loss: {loss_vector[-1][1]:.4f}, acc: {accuracy}')
        if i % (len(train_loader) // 4) == 0 or i + 1 == len(train_loader):
            validate(val_loss_vector, val_accuracy_vector, x)

def validate(loss_vector, accuracy_vector, epoch):
    model.eval()
    val_loss, correct = 0, 0
    for i, batch in enumerate(val_loader):
        for key in batch.keys():
            batch[key] = batch[key].to(device)
        val_output = model(**batch)
        val_loss += val_output.loss.item()

        pred = torch.argmax(val_output.logits, dim=1)

        correct += pred.eq(batch['labels']).sum()

    loss_vector.append((epoch, val_loss/len(val_loader)))

    accuracy = 100. * correct.to(torch.float32) / len(val_loader.dataset)
    accuracy_vector.append((epoch, accuracy.item()))
    print(f'[epoch {epoch:.2f}] validation loss: {loss_vector[-1][1]:.4f}, acc: {accuracy}')

train_loss_vec = []
train_acc_vec = []
val_loss_vec = []
val_acc_vec = []

for epoch in range(5):
    train(train_loss_vec, train_acc_vec, val_loss_vec, val_acc_vec, epoch)
    mkdir(f"./{output_dir}/e{epoch}")
    model.save_pretrained(f"./{output_dir}/e{epoch}")
    tokenizer.save_pretrained(f"./{output_dir}/e{epoch}")
    # validate(val_loss_vec, val_acc_vec, epoch)

logs = {}
logs['train_loss'] = train_loss_vec
logs['train_acc'] = train_acc_vec
logs['val_loss'] = val_loss_vec
logs['val_acc'] = val_acc_vec
for i in logs.values():
    print(i)
with open(f"./{output_dir}/logs.json", 'w') as f:
    f.write(json.dumps(logs, indent=2))