#%%
import argparse

#%% parse arguments
parser = argparse.ArgumentParser(description='Train sentence transfomer')
parser.add_argument('-m', '--model', type=str, help='base model name or path', required=True)
parser.add_argument('-d', '--dense', action='store_true', help='whether to add dense layer')
parser.add_argument('-a', '--activation', type=str, help='dense layer activation', default='tanh')
parser.add_argument('-w', '--warmup', type=int, help='warmup step', default=300)
parser.add_argument('-o', '--output', type=str, help='output name', required=True)
parser.add_argument('-e', '--epoch', type=int, help='no. of epoch', default=5)

args = parser.parse_args()

print(f'Arguments: {args}')
#%%
print("Importing...")
from sentence_transformers import SentenceTransformer, models, InputExample, losses, evaluation
from torch.utils.data import DataLoader
from torch import nn
import random
import copy
import postgres
import os
from sklearn.model_selection import train_test_split
random.seed(0)

#%%
print("Setting model...")
modules = []
word_embedding_model = models.Transformer(args.model, max_seq_length=128)
modules.append(word_embedding_model)

pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode_cls_token=True, pooling_mode_mean_tokens=False)
modules.append(pooling_model)

if args.dense:
    if args.activation == 'tanh':
        dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.Tanh())
    elif args.activation == 'sigmod':
        dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.Sigmoid())
    elif args.activation == 'relu':
        dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.ReLU())
    assert dense_model, f"unknown activation function {args.activation}"
    modules.append(dense_model)

model = SentenceTransformer(modules=modules)
print(model)
#%% Dataset
print("Get data...")
data = postgres.query("""
SELECT cleaned_content, cluster_uid FROM comment
WHERE cluster_uid IS NOT null
ORDER BY comment_id
""", return_dict=True)

# AND cluster_uid != 0
#%%
cluster_uid = list(set([i['cluster_uid'] for i in data]))
train_cluster, test_cluster = train_test_split(cluster_uid, train_size=0.8, random_state=0)
# test_cluster = [22, 32, 28, 4, 19, 30, 33, 7, 10]
# train_cluster = [35, 26, 23, 13, 16, 39, 38, 6, 5, 11, 40, 9, 20, 24, 0, 1, 21, 18, 14, 37, 2, 8, 29, 36, 34, 31, 3, 17, 27, 25, 12, 15]
# train_cluster = [i for i in range(n_cluster) if i not in test_cluster]
grouped = {}
grouped_test = {}
for i in train_cluster:
    grouped[i] = [sentence['cleaned_content'] for sentence in data if i == sentence['cluster_uid']]

for i in test_cluster:
    grouped_test[i] = [sentence['cleaned_content'] for sentence in data if i == sentence['cluster_uid']]

print(f"Train cluster: {train_cluster}")
print(f"Test cluster: {test_cluster}")
#%%
print("Format dataset...")
def generate_dataset(_grouped):

    def all_same(_sentences):
        for i in _sentences:
            for j in _sentences:
                if i != j:
                    return False
        return True

    def duplicate(data, k=1):
        for cluster_id, _sentences in data.items():
            tmp = copy.deepcopy(_sentences)
            for i in range(k):
                _sentences += tmp

    dataset = []
    same = copy.deepcopy(_grouped)
    diff = copy.deepcopy(_grouped)
    other = copy.deepcopy(_grouped)
    duplicate(same, k=1) # duplicate for balanced dataset

    for cluster_id, sentences in same.items():
        while len(sentences) > 1:
            if all_same(sentences):
                break
            choices = random.choices(sentences, k=2)
            if choices[0] != choices[1]:
                dataset.append(InputExample(texts=[choices[0], choices[1]], label=1.0))
                sentences.remove(choices[0])
                sentences.remove(choices[1])

    for cluster_id, sentences in diff.items():
        other_cluster = [value for key, value in other.items() if key != cluster_id]
        other_cluster = [item for sublist in other_cluster for item in sublist] # flatten lists
        for sentence in sentences:
            choice = random.choice(other_cluster)
            if choice != choices[1]:
                other_cluster.remove(choice)
                dataset.append(InputExample(texts=[sentence, choice], label=0.0))

    print(f"Dataset length: {len(dataset)}")
    return dataset

#%%
train_cluster_dataset = generate_dataset(grouped)
test_cluster_dataset = generate_dataset(grouped_test)

# train_dataset, test_dataset = train_test_split(dataset, train_size=0.8)
train_dataset = train_cluster_dataset
test_dataset = test_cluster_dataset
train_dataloader = DataLoader(train_cluster_dataset, shuffle=True, batch_size=16)
evaluator = evaluation.EmbeddingSimilarityEvaluator(
    [i.texts[0] for i in test_dataset],
    [i.texts[1] for i in test_dataset],
    [i.label for i in test_dataset])

# %%
train_loss = losses.CosineSimilarityLoss(model)

#%%
output_dir = args.output
try:
    os.mkdir(output_dir)
except:
    pass

try:
    os.mkdir(output_dir + '/best')
except:
    pass

def callback(score, epoch, steps):
    print(f'[epoch {epoch}] score: {score*10:.2f}')
    if epoch == 2 and steps == -1:
        raise Exception("this is the end")
print("Train model...")
# model.to('cuda:0')
model.fit(  train_objectives=[(train_dataloader, train_loss)],
            epochs=args.epoch,
            evaluator=evaluator,
            evaluation_steps=50,
            callback=callback,
            output_path=output_dir + '/best',
            save_best_model=True,
            show_progress_bar=False,
            warmup_steps=args.warmup,
            optimizer_params={'lr': 2e-6}
            )

print("Save model...")
model.save(output_dir)
print(f"Model saved: {output_dir}")
print("-"*30)
