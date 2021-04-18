#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn import cluster
from sklearn.decomposition import PCA as sklearnPCA

from sentence_transformers import SentenceTransformer

from tqdm import tqdm
import json

#%%
def elbow(data, output_dir=None, mini=3, maxi=20, steps=1, show_graph=False, save=True):
    """Generate graph of elbow method.

    Args:
        data: np 2darray in size (number of sequence x number of dimension)
        output_dir(str) : output directory of the file
        mini(int): minimum k to test
        maxi(int): maximum k to test
        show_graph(bool): whether show will be called
        save(bool): whether the file will be saved
    """
    # data = np.array(torch.load(x+'_output_post_1202_v1.pt', map_location=map_location).cpu())
    print("Input data count = {}, features : {}".format(data.shape[0], data.shape[1]))

    # k means determine k
    distortions = []
    k_range = range(mini, maxi, steps)
    for k in tqdm(k_range):
        kmeanModel = cluster.KMeans(n_clusters=k).fit(data)
        kmeanModel.fit(data)
        d = sum(np.min(cdist(data, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0]
        print(d)
        distortions.append(d)
    # Plot the elbow
    fig2 = plt.figure(1)
    plt.plot(k_range, distortions, 'b-')
    plt.xlabel('k')
    plt.ylabel('distortion')
    plt.title('Elbow Method on 2021-01-02 post')
    if show_graph:
        plt.show()
    if save:
        fig2.savefig(f"{output_dir}/elbow.png")
        print(f"File saved at {output_dir}/elbow.png")

#%%
if __name__ == '__main__':
    with open('data/2021-01-05_comments.json', 'r') as f:
        sequences = json.loads(f.read())
    sentence_transformer = SentenceTransformer('./models/lihkgBERT-sentenceTransformer-CLS')
    sentence_transformer.to("cuda:0")
    class_vector = sentence_transformer.encode(sequences)
    params = {
        'output_dir': 'output/report/comments_20',
        'save': True,
        'mini': 10,
        'maxi': 70,
        'steps': 5
    }
    elbow(class_vector, **params)
# %%
