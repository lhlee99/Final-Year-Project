import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch

from sklearn import cluster
from scipy.spatial.distance import cdist

from sklearn.decomposition import PCA as sklearnPCA


if __name__ == "__main__":

    print("Reading model_data.pt ...")
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    print(map_location)

    x="wwm"
    data = np.array(torch.load(x+'_output_post_1202_v1.pt', map_location=map_location).cpu())
    
    print("Input data count = {}, features : {}".format(data.shape[0], data.shape[1]));
    
    print("Determining k ....");
    
    y=10
    # k means determine k
    distortions = []
    k_range = range(3,y)
    for k in k_range:
        kmeanModel = cluster.KMeans(n_clusters=k).fit(data)
        kmeanModel.fit(data)
        distortions.append(sum(np.min(cdist(data, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])
    # Plot the elbow
    fig2 = plt.figure(1)
    plt.plot(k_range, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    #plt.show()
    fig2.savefig('./' + "elbow_" + x +"_"+ str(y) + ".png")
