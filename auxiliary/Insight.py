import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
from tqdm.notebook import tqdm


def optimize_n_clusters(data, r=(2, 50), plot=True):
    """
    Runs knee-location to determine the most efficient value of n_clusters for KMeans.

    Parameters:
        data: (csr_matrix): data for clustering
        r: (tuple(int, int)): range of values to iterate over
        plot: (bool): whether or not to plot the results

    Returns:
        knee.knee: (int): optimal value of n_clusters

    """
    K = range(r[0], r[1])
    inertia = []
    for k in tqdm(K):
        inertia.append(KMeans(n_clusters=k).fit(data).inertia_)

    i = np.arange(len(inertia))
    knee = KneeLocator(i, inertia, S=1, curve='convex', direction='decreasing', interp_method='polynomial')

    if plot:
        fig = plt.figure(figsize=(5, 5))
        knee.plot_knee()
        plt.title('Elbow Method')
        plt.xlabel('cluster numbers')
        plt.ylabel("Inertia")
        plt.show()

    return knee.knee


def cluster_sizes(labels):
    """
    Quantitative analysis function for final clustering labels

    Parameters:
        labels: (np.array): clustering labels

    Returns:
        d: (dict): dictionary of labels and quantities of entries, corresponding to each label

    """
    d = {}
    for l in labels:
        if not d.get(l):
            d[l] = 1
        else:
            d[l] += 1
    return d


def get_top_features(data, labels, cluster: int, names):
    """
    Presents top 6 most meaningful features among the chosen cluster according to TF-IDF

    Parameters:
        data: (csr_matrix): sparse matrix of values from TF-IDF vectorizer
        labels: (np.array): array of clustering, result from running a clustering algorithm
        cluster: (int): chosen cluster to get the top features from
        names: (np.array): array that allows to retrieve names from digital feature values

    Returns:
        res: (list): list of names of top features among the chosen cluster

    """
    tfidf_sorting = np.argsort(np.sum(data[labels == cluster].toarray(), axis=0))[::-1]
    res = []
    for i in tfidf_sorting[:6]:
        res.append(names[i])
    return res
