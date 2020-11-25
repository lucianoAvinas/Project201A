import tifffile
import os
import pandas
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from matplotlib.colors import ListedColormap
from sklearn.metrics import silhouette_score, calinski_harabasz_score, \
                            davies_bouldin_score


class DataPrep:
    # assumes overview.csv is in root folder
    def __init__(self, data_dir, test_frac=0.20, seed=48):
        # filetype (str): choices ['tiff', 'npy']
        csv_dat = pandas.read_csv('overview.csv')
        n = len(csv_dat)

        contrast_id = np.array(csv_dat['Contrast'])
        filetype = (os.listdir(data_dir)[0]).split('.')[-1]

        if 'tif' in filetype:
            file_paths = [None]*n
            tiff_names = csv_dat['tiff_name']
            for i in range(n):
                file_paths[i] = os.path.join(data_dir, tiff_names[i])

            im_stack = tifffile.imread(file_paths)
        else:
            im_stack = np.load(os.path.join(data_dir, 'all_data.npy'))
            im_stack = im_stack.reshape(n, -1, 1)
        

        np.random.seed(seed)
        perm_inds = np.random.permutation(range(n))

        self.data = im_stack[perm_inds,:,:]
        self.out_ids = contrast_id[perm_inds]
        self.test_len = int(n * test_frac)
        self.N = n

    def get_test(self):
        data_struct = {'train':None, 'test':None}

        # in case test_len = 0, supply None
        data_struct['train'] = (self.data[:-self.test_len or None,:,:], 
                                self.out_ids[:-self.test_len or None])
        if self.test_len == 0:
            data_struct['test'] = (np.array([]), np.array([]))
        else:
            data_struct['test'] = (self.data[-self.test_len:,:,:], 
                                    self.out_ids[-self.test_len:])
        return data_struct

    def get_kfold(self, k):
        # in case test_len = 0, supply None
        data_subset = self.data[:-self.test_len or None,:,:]
        out_subset = self.out_ids[:-self.test_len or None]
        data_struct = {'train':None, 'valid':None}

        train_inds = np.full(self.N - self.test_len, True)
        val_len = (self.N - self.test_len)//k
        for i in range(k):
            train_inds[i*val_len:(i+1)*val_len] = False

            data_struct['train'] = (data_subset[train_inds,:,:], 
                                    out_subset[train_inds])
            data_struct['valid'] = (data_subset[~train_inds,:,:], 
                                    out_subset[~train_inds])
            train_inds[i*val_len:(i+1)*val_len] = True

            yield data_struct


def TSNE_wrapper(data, perplexity, early_exaggeration, learning_rate, n_iter):
    tsne_obj = TSNE(perplexity=perplexity, early_exaggeration=early_exaggeration,
                    learning_rate=learning_rate, n_iter=n_iter,
                    n_iter_without_progress=n_iter, method='exact')
    # Acceptable value ranges:
    # perplexity \in [5, 50], early_exaggeration \in [1, 100]
    # learning_rate \in [10, 1000], n_iter \in [250, 2500]
    return tsne_obj.fit_transform(normalize(data.reshape(data.shape[0], -1)))


def cluster_calcs(X_embed, y):
    intra_l2 = [None, None]

    X_true = X_embed[y]
    true_intra, mean_true = intra_calc(X_true)
    intra_l2[0] = true_intra

    X_false = X_embed[~y]
    false_intra, mean_false = intra_calc(X_false)
    intra_l2[1] = false_intra

    diff = mean_true - mean_false
    inter_l2 = np.sum(diff*diff)

    return intra_l2, inter_l2


def intra_calc(X):
    n = X.shape[0]
    X_mean = X.mean(axis=0)
    diff = X-X_mean

    return 2*n*np.sum(diff*diff), X_mean


def plot_embedding(X_embed, y):
    # blue is no contrast added, red is contrast added
    plt.scatter(X_embed[:,0], X_embed[:,1], c=y, cmap=ListedColormap(['blue', 'red']))
    plt.show()


def default_cluster_stats(data_dir):
    data = DataPrep(data_dir)

    all_train = data.get_test()['train']
    X_embed = TSNE_wrapper(all_train[0], 30, 12, 200, 1000)
    cluster_stats = cluster_calcs(X_embed, all_train[1])

    return cluster_calcs

def all_metric_evals():
    # We have ground truths but no predicted clusters
    # To keep it simple I will just use the no groundtruth metic evaluations
    metrics = {'Silhoutte': silhouette_score, 
               'Calinski': calinski_harabasz_score, 
               'Davies': davies_bouldin_score}

    # defined in preprocess.produce_all_datasets
    sizes = [8, 16, 32, 64, 128]
    dims = [8, 16, 32, 64, 100]
    n1 = len(sizes)
    n2 = len(dims)

    ticks = sorted(list(set(sizes) | set(dims)))

    for name, func in metrics.items():
        resample = [None]*n1
        for i in range(n1):
            data_dir = os.path.join('resampled_tiffs', str(sizes[i]))
            data = DataPrep(data_dir).get_test()['train']
            X_embed = TSNE_wrapper(data[0], 30, 12, 200, 1000)

            resample[i] = func(X_embed, data[1])

        dimreduc = [None]*n2
        for i in range(n2):
            data_dir = os.path.join('dimension_reduced', str(dims[i]))
            data = DataPrep(data_dir).get_test()['train']
            X_embed = TSNE_wrapper(data[0], 30, 12, 200, 1000)

            dimreduc[i] = func(X_embed, data[1])

        subtitle = ' (Higher is Better)'
        if name == 'Davies':
            subtitle = ' (Lower is Better)'

        plt.plot(sizes, resample, label='Downsampled')
        plt.plot(dims, dimreduc, label='PCA-Reduced')
        plt.xlabel('Dimension or Sidelength')
        plt.ylabel('Metric Score')
        plt.title(name+subtitle)
        plt.legend()
        plt.xticks(ticks, ticks)

        plt.savefig(name+'.png')
        plt.cla()
        plt.clf()
