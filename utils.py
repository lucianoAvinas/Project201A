import tifffile
import os
import pandas
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
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
                    learning_rate=learning_rate, n_iter=int(n_iter),
                    n_iter_without_progress=n_iter, min_grad_norm=0, method='exact')
    # Acceptable value ranges:
    # perplexity \in [5, 50], early_exaggeration \in [1, 100]
    # learning_rate \in [10, 1000], n_iter \in [250, 2500]
    return tsne_obj.fit_transform(data.reshape(data.shape[0], -1))


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


def class_acc(X_embed, y):
    kmeans_obj = KMeans(2)
    y_pred = kmeans_obj.fit_predict(X_embed).astype(bool)

    # labels may be permuted
    y_pred = ~y_pred if np.mean(y[y_pred]) < 0.5 else y_pred
    return np.sum(y_pred == y)/len(y)


def intra_calc(X):
    n = X.shape[0]
    X_mean = X.mean(axis=0)
    diff = X-X_mean

    return 2*n*np.sum(diff*diff), X_mean


def varied_tsne():
    X, y = DataPrep(os.path.join('resampled_tiffs', '64')).get_test()['train']
    for i in range(4):
        X_embed = TSNE_wrapper(X, 30, 12, 200, 1000)

        plt.rcParams["figure.figsize"] = (14, 14)
        plt.rcParams.update({'font.size': 22})
        sc = plt.scatter(X_embed[:,0], X_embed[:,1], s=100, c=y, cmap=ListedColormap(['blue', 'red']))
        L = plt.legend(*sc.legend_elements())
        L.get_texts()[0].set_text('No Contrast')
        L.get_texts()[1].set_text('Contrast Added')
        plt.savefig(os.path.join('presentation_imgs', f'tsne_rep{i}.png'), dpi=400)
        plt.cla(), plt.clf()



def save_ct():
    table = pandas.read_csv('overview.csv')

    noc_img = tifffile.imread(os.path.join('tiff_images',table['tiff_name'][5]))
    c_img = tifffile.imread(os.path.join('tiff_images',table['tiff_name'][57]))
    noc_img_64 = tifffile.imread(os.path.join('resampled_tiffs','64',table['tiff_name'][5]))
    c_img_64 = tifffile.imread(os.path.join('resampled_tiffs','64',table['tiff_name'][57]))

    vmax = max(np.max(noc_img), np.max(c_img), np.max(noc_img_64), np.max(c_img_64))
    vmin = min(np.min(noc_img), np.min(c_img), np.min(noc_img_64), np.min(c_img_64))

    names = ['no_contrast.png', 'contrast.png', 'no_contrast_64.png', 'contrast_64.png']
    imgs = [noc_img, c_img, noc_img_64, c_img_64]
    plt.rcParams["figure.figsize"] = (14, 14)
    for i in range(4):
        plt.imshow(imgs[i], plt.cm.gray, vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.savefig(os.path.join('presentation_imgs', names[i]), dpi=400)
        plt.cla(), plt.clf()


def plot_embedding(X_embed, y):
    # blue is no contrast added, red is contrast added
    plt.rcParams["figure.figsize"] = (14, 14)
    plt.rcParams.update({'font.size': 22})
    sc = plt.scatter(X_embed[:,0], X_embed[:,1], s=100, c=y, cmap=ListedColormap(['blue', 'red']))
    L = plt.legend(*sc.legend_elements())
    L.get_texts()[0].set_text('No Contrast')
    L.get_texts()[1].set_text('Contrast Added')
    plt.savefig(os.path.join('presentation_imgs', 'true_labels.png'), dpi=400)
    plt.cla(), plt.clf()

    kmeans_obj = KMeans(2)
    y_pred = kmeans_obj.fit_predict(X_embed).astype(bool)
    y_pred = ~y_pred if np.mean(y[y_pred]) < 0.5 else y_pred

    sc = plt.scatter(X_embed[:,0], X_embed[:,1], s=100, c=y_pred, cmap=ListedColormap(['blue', 'red']))
    L = plt.legend(*sc.legend_elements())
    L.get_texts()[0].set_text('No Contrast')
    L.get_texts()[1].set_text('Contrast Added')
    plt.savefig(os.path.join('presentation_imgs', 'pred_labels.png'), dpi=400)
    plt.cla(), plt.clf()

    print('Pred Acc:', np.sum(y_pred == y)/len(y))


def default_cluster_stats(data_dir):
    data = DataPrep(data_dir)

    all_train = data.get_test()['train']
    X_embed = TSNE_wrapper(all_train[0], 30, 12, 200, 1000)
    cluster_stats = cluster_calcs(X_embed, all_train[1])

    return cluster_stats

def all_metric_evals():
    # We have ground truths but no predicted clusters
    # To keep it simple I will just use the no groundtruth metic evaluations
    metrics = [silhouette_score, calinski_harabasz_score, 
                davies_bouldin_score, class_acc]
    metric_names = ['Silhoutte', 'Calinski', 'Davies', 'Accuracy']

    # defined in preprocess.produce_all_datasets
    sizes = [8, 16, 32, 64, 128]
    dims = [8, 16, 32, 64, 100]
    n1 = len(sizes)
    n2 = len(dims)

    ticks = sorted(list(set(sizes) | set(dims)))

    resample = np.zeros((len(metrics), n1))
    for i in range(n1):
        data_dir = os.path.join('resampled_tiffs', str(sizes[i]))
        data = DataPrep(data_dir).get_test()['train']
        X_embed = TSNE_wrapper(data[0], 30, 12, 200, 1000)

        for k in range(len(metrics)):
            resample[k, i] = metrics[k](X_embed, data[1])

    dimreduc = np.zeros((len(metrics), n2))
    for i in range(n2):
        data_dir = os.path.join('dimension_reduced', str(dims[i]))
        data = DataPrep(data_dir).get_test()['train']
        X_embed = TSNE_wrapper(data[0], 30, 12, 200, 1000)

        for k in range(len(metrics)):
            dimreduc[k, i] = metrics[k](X_embed, data[1])

    for k in range(len(metrics)):
        name = metric_names[k]

        subtitle = ' (Higher is Better)'
        if name == 'Davies':
            subtitle = ' (Lower is Better)'

        plt.plot(sizes, resample[k], label='Downsampled')
        plt.plot(dims, dimreduc[k], label='PCA-Reduced')
        plt.xlabel('Dimension or Sidelength')
        plt.ylabel('Metric Score')
        plt.title(name+subtitle)
        plt.legend()
        plt.xticks(ticks, ticks)

        plt.savefig(name+'.png')
        plt.cla()
        plt.clf()
        
def calculate_response(data, labels, perplexity, early_exaggeration, learning_rate, n_iter):
    metrics = [silhouette_score, calinski_harabasz_score, 
                davies_bouldin_score, class_acc]

    response = np.zeros(len(metrics))
    X_embed = TSNE_wrapper(data, perplexity, early_exaggeration, learning_rate, n_iter)
    
    for i in range(len(metrics)):
        response[i] = metrics[i](X_embed, labels)
    return response
