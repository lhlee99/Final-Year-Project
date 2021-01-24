import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
from sklearn import cluster
import hdbscan
from sklearn.decomposition import PCA as sklearnPCA
import json

#matplotlib inline
sns.set_context('poster')
sns.set_color_codes()
plotKwds = {'alpha' : 0.25, 's' : 5, 'linewidths':0}

outputDir = "./report"
try:
    os.mkdir(outputDir)
except Exception:
    pass

def reduce_dim(centroids, dataframe):
    
    print("Reducing dimension ...");
    dataSize = dataframe.shape[0]
    try:
        mergeData = np.concatenate((dataframe, centroids), axis=0)
    except Exception:
        mergeData = dataframe

    recNorm = (mergeData - mergeData.min())/(mergeData.max() - mergeData.min())
    pca = sklearnPCA(n_components=5)
    point = pca.fit_transform(recNorm)
    
    #[center_pt, data_pt]
    return [point[dataSize:, :], point[: dataSize, :] ]

def draw_graph(algorithm, centroids, labels, dataframe):
    
    [centerPoints, dataPoints] =  reduce_dim(centroids, dataframe)
    clusterCnt = labels.max() + 1

    print("Drawing clustered result ...");
    fig = plt.figure(1)
    plt.suptitle('Cluserters found by ' + algorithm, fontsize=16)

    x = dataPoints[:, 0]
    y = dataPoints[:, 1]
    scatter = plt.scatter(x, y, c = labels + 1, **plotKwds)
    
    if clusterCnt < 11:
        for ctIdx in range(0, clusterCnt):
            x = centerPoints[ctIdx, 0]
            y = centerPoints[ctIdx, 1]

            plt.annotate(str(ctIdx + 1), 
                    [x, y],
                    horizontalalignment='center',
                    verticalalignment='center',
                    size=5, weight='bold',
                    color='white',
                    backgroundcolor = scatter.to_rgba(ctIdx + 1)) 
    cb = plt.colorbar(scatter)
    cb.ax.tick_params(labelsize=10)
    #plt.gca().set_facecolor('xkcd:salmon')
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.xlabel('pca1', fontsize=9)
    plt.ylabel('pca2', fontsize=9)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    #fig.tight_layout()
    plt.show()
    fig.savefig(outputDir + '/' + algorithm + ".png");

def do_clustering(data, k ,algorithm, args, kwds):
    
    print("Clustering ", str(algorithm.__name__));
    clusterModel = algorithm(*args, **kwds)
    # compute cluster centers and predict cluster index for each sample
    clusterLabels = clusterModel.fit_predict(data)
    centers = []
    
    clusterCount = clusterLabels.max() + 1
    if clusterCount > 0:
        try:
            centers = clusterModel.cluster_centers_
        except Exception:
            rec_count = np.zeros(clusterCount)
            centers = np.zeros((clusterCount, data.shape[1]))
            for (i, label) in enumerate(clusterLabels):
                centers[label] += data[i]
                rec_count[label] += 1
            for (i, count) in enumerate(rec_count):
                if count != 0:
                    centers[i] /= count
    
    draw_graph(str(algorithm.__name__), centers, clusterLabels, data)

    fileName = outputDir + '/' + str(algorithm.__name__) + '.json'
    print("Writing result to {} ...".format(fileName));
    
    output = {}
    with open(fileName, 'w') as f:
        if k != 0:
            output['numberOfCluster'] = float(clusterCount)
            if len(centers):
                formattedCenters = [[np.round(float(i), 2) for i in nested] for nested in centers]
                output['centerVector'] = formattedCenters
        labelList = clusterLabels.tolist()
        adjustedLabelList = [x+1 for x in labelList]
        output['clusterLabel'] = adjustedLabelList
        f.write(json.dumps(output))

    print("Clustering finished ", str(algorithm.__name__) + "\n");

if __name__ == "__main__":
    
    print("Reading data.pt ...")
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    data = np.array(torch.load('output/wwm_output_1202_v5.pt', map_location=map_location))[:, :]
    print("Input data count = {}, features : {}".format(data.shape[0], data.shape[1]));
    
    while True:   
        print("0. Quit")     
        print("1. kMean")
        print("2. AgglomerativeClustering(slow)")
        print("3. MiniBatchKMeans")

        method = int(input("Input clustering method in number: "));
        #method = 1 
        if method == 1:
            k = int(input("Input cluster count : "))
            #k = 4
            do_clustering(data, k, cluster.KMeans, (), {'n_clusters':k})
        elif method == 2:
            k = int(input("Input cluster count : "))
            do_clustering(data, k, cluster.AgglomerativeClustering, (), {'n_clusters':k, 'linkage':'ward'})
        elif method == 3:
            k = int(input("Input cluster count : "))
            do_clustering(data, k, cluster.MiniBatchKMeans, (), {'n_clusters':k})
        elif method == 0:
            break;
