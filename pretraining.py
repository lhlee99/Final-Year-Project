#%%
from transformers import BertForPreTraining, DataCollatorForLanguageModeling, TextDatasetForNextSentencePrediction, AutoTokenizer, Trainer, TrainingArguments
import postgres
from tqdm import tqdm
import re
import torch
#%%
def actual_length(sequence):
    tmp = re.sub(r"[A-Za-z']+", "_", sequence)
    return len(tmp)
comments = postgres.query("""
SELECT cleaned_content FROM comment
ORDER BY comment_id
LIMIT 1000
""")
comments = [i[0].lower() for i in comments if actual_length(i[0]) >= 5]

# %%
from utils import split_text
splitted_comments = [split_text(i, 40, recursive_until=128) for i in comments]

# %%
from sklearn.model_selection import train_test_split
train, test = train_test_split(splitted_comments, test_size=0.2, random_state=0)
# %%
with open("train_dataset.txt", 'w') as f:
    tmp = '\n\n'.join(['\n'.join(i) for i in train])
    f.write(tmp + '\n')
with open("test_dataset.txt", 'w') as f:
    tmp = '\n\n'.join(['\n'.join(i) for i in test])
    f.write(tmp + '\n')
# %%
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = BertForPreTraining.from_pretrained("bert-base-chinese")

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
new_tokens = [i[1] for i in freq if i[0] >= 5]
print(tokenizer.add_tokens(new_tokens))
model.resize_token_embeddings(len(tokenizer))

#%%
train_dataset = TextDatasetForNextSentencePrediction(tokenizer=tokenizer, file_path='./train_dataset.txt', block_size=128)
# test_dataset = TextDatasetForNextSentencePrediction(tokenizer=tokenizer, file_path='./test_dataset.txt', block_size=128)

# %%
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)
# %%
training_args = TrainingArguments(
    output_dir="./TestPreTrainingBERT",
    overwrite_output_dir=True,
    num_train_epochs=2,
    # save_steps=10_000,
    save_total_limit=2,
    logging_steps=5,
    # prediction_loss_only=True,
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset[:50],
    # eval_dataset=test_dataset,
)
# %%
%%time
trainer.train()
# %%
trainer.save_model("./TestPreTrainingBERT")
tokenizer.save_pretrained("./TestPreTrainingBERT")

#%%
trainer.evaluate()

#%% COMPARASION
base_tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
base_data_collator = DataCollatorForLanguageModeling(tokenizer=base_tokenizer)
base_test_dataset = TextDatasetForNextSentencePrediction(tokenizer=base_tokenizer, file_path='./test_dataset.txt', block_size=128)
base_model = BertForPreTraining.from_pretrained("bert-base-chinese")

#%%
base_trainer = Trainer(
    model=base_model,
    data_collator=base_data_collator,
    eval_dataset=base_test_dataset,
)
#%%
base_trainer.evaluate()
# %%

# %%

training_args = TrainingArguments(
    output_dir="./lihkgBERT",
    overwrite_output_dir=True,
    num_train_epochs=3,
    # save_total_limit=5,
    # prediction_loss_only=True,
    save_strategy='epoch',
    seed=42
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    # eval_dataset=test_dataset,
)
# %%
