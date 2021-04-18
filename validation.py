#%% INIT
print("Importing")
import json
import os
from utils import split_text, generate_class_vector
import clustering
from sklearn import metrics
import argparse
from sentence_transformers import SentenceTransformer

#%% parse arguments
parser = argparse.ArgumentParser(description='Do validation')
parser.add_argument('-i', '--input', type=str, help='dataset location', required=True)
parser.add_argument('-o', '--output', type=str, help='output location', default=None)
args = parser.parse_args()
#%%
print("Loading dataset")
def normalize_labels(_labels):
    options = list(set(_labels))
    normalized = []
    for i in _labels:
        normalized.append(options.index(i))
    return normalized

with open(args.input, 'r') as f:
    data = json.loads(f.read())
    sequences = data['comments']
    labels_true = normalize_labels(data['labels'])

#%%
if args.output:
    try:
        os.mkdir(args.output)
    except:
        pass

models = [
    {
        'model_name': 'bert-base-chinese',
        'abbr': 'base',
    },
    {
        'model_name': './models/lihkgBERT-sentenceTransformer',
        'abbr': 'lihkgBERT-st-CLS',
        'st': True
    }
]
def print_score(*args):
    ri = metrics.rand_score(*args)
    ami = metrics.adjusted_mutual_info_score(*args)
    h = metrics.homogeneity_score(*args)
    c = metrics.completeness_score(*args)
    v = metrics.v_measure_score(*args)
    print([ri, ami, h, c, v])

SAVE = args.output != None
max_cluster = max(labels_true) if 0 not in labels_true else max(labels_true) + 1
for model in models:
    if 'st' in model:
        sentence_transformer = SentenceTransformer(model['model_name'])
        class_vector = sentence_transformer.encode(sequences)
    else:
        class_vector = generate_class_vector(sequences,
                                                model_name=model['model_name'],
                                                output_name=f"{args.output}/{model['abbr']}.pt",
                                                roberta=bool('roberta' in model),
                                                # save=SAVE)
                                                save=False)

    model['labels_kmean'] = clustering.clusteringf('k-mean', class_vector, max_cluster, seed=0, rtn='label',
                                    outputDir=f"{args.output}/{model['abbr']}", save=SAVE)

    print('kmean: ', end='')
    print_score(labels_true, model['labels_kmean'])

# %% EVALUATION
print("Summary")
print("(ri, ami, h_score, c_score, v_measure)")
for model in models:
    print(model['model_name'])
    print('-'*20)
    print_score(labels_true, model['labels_kmean'])
    print('')