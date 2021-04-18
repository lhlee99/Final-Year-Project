#%% parse arguments
import argparse
parser = argparse.ArgumentParser(description='Do validation')
parser.add_argument('-i', '--input', type=str, help='dataset location', required=True)
parser.add_argument('-o', '--output', type=str, help='output location', default=None)
parser.add_argument('-k', '--k-cluster', type=int, help='number of cluster', default=None, required=True)
args = parser.parse_args()


#%% INIT
print("Importing")
import json
import os
from utils import split_text, generate_class_vector
import clustering
from sklearn import metrics
from sentence_transformers import SentenceTransformer
import topic_modeling

#%%
print("Loading dataset")

with open(args.input, 'r') as f:
    sequences = json.loads(f.read())

#%%
if args.output:
    try:
        os.mkdir(args.output)
    except:
        pass

models = [
    {
        'model_name': './models/lihkgBERT-sentenceTransformer',
        'abbr': 'lihkgBERT-st',
    }
]

#%%
SAVE = args.output != None
n_cluster = args.k_cluster
for model in models:
    sentence_transformer = SentenceTransformer(model['model_name'])
    class_vector = sentence_transformer.encode(sequences)

    model['labels_kmean'] = clustering.clusteringf('k-mean', class_vector, n_cluster, seed=0, rtn='label',
                                    outputDir=f"{args.output}/kmean", save=SAVE)

    model['kmean_topics'] = topic_modeling.termFrequency(sequences, model['labels_kmean'], n_cluster,
                                                            output_dir=f"{args.output}/kmean", save=SAVE)



#%%
def print_topics(topics, distribution):
    for key, value in topics.items():
        print(f"Topic {key}: {distribution[distribution.Topic == key].Size.values[0]}\n{'-'*20}")
        if len(value) == 20:
            idx = 0
            for i in range(4):
                for j in range(5):
                    to_print = f"{value[idx][0]} [{value[idx][1]:.4f}]"
                    idx += 1
                    print(f'{to_print:<20}', end='')
                print('')
        print('')

print(f"Summary\n{'-'*50}")
for model in models:
    print(model['model_name'])
    print('-'*40)
    print_topics(*model['kmean_topics'])

# %%
