import numpy as np
import time
import utils
import os
import matplotlib.pyplot as plt

from scipy.stats import probplot
from preprocess import to_design, to_01
from multiprocessing import Pool
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C

MATERN_ELL = 2.718492
MATERN_SIG2 = 9.232834e-04

RBF_ELL = 2.396396
RBF_SIG2 = 9.26446e-04

S2_TOT = 0.00239413

def get_response(design, data):
    np.random.seed()
    X, y = data
    response = np.zeros(design.shape[0])

    for i in range(len(response)):
        pnt = design[i]
        X_embed = utils.TSNE_wrapper(X, *pnt)
        response[i] = utils.class_acc(X_embed, y)

    return response


def fit_gp(design, response, kern='matern'):
    if kern == 'matern':
        kernel = C(MATERN_SIG2) * Matern(MATERN_ELL, nu=5/2)
        acc_predictor = GaussianProcessRegressor(kernel=kernel, alpha=1*(S2_TOT-MATERN_SIG2))
    elif kern == 'rbf':
        kernel = C(RBF_SIG2) * RBF(RBF_ELL)
        acc_predictor = GaussianProcessRegressor(kernel=kernel, alpha=S2_TOT-RBF_SIG2)
    else:
        raise Exception(f'{kern} not supported')

    acc_predictor.fit(to_01(design), response)
    return acc_predictor


def pred_gp(gp, new_design):
    if np.max(new_design) < 1:
        # int casting
        new_design = to_design(new_design)

    return gp.predict(to_01(new_design))


def TSNEvar_multi(split, threads, reps):
    if __name__ == '__main__':
        st = time.time()
        design, data = setup_data(split)
        arrs = []
        def collector(result):
            arrs.append(result)

        worker_pool = Pool(threads) 
        for i in range(reps):
            worker_pool.apply_async(get_response, args=(design, data), callback=collector)
        worker_pool.close()
        worker_pool.join()

        en = time.time()
        print((en-st)/60)

        coll_arr = np.vstack(arrs)
        return coll_arr.var(ddof=1)


def setup_data(split):
    data_obj = utils.DataPrep(os.path.join('resampled_tiffs', '64'))
    if split == 'train':
        data = data_obj.get_test()['train']
    elif split == 'all':
        data = (data_obj.data, data_obj.out_ids)
    else:
        raise Exception(f'{split} is not a supported split')

    design = np.load(os.path.join('experiment_data', 'design_points.npy'))
    
    return design, data


def resample_data(split, resample=0):
    design, data = setup_data(split)
    if resample > 0:
        design = to_design(np.random.random((resample, 4)))        

    response = get_response(design, data)

    return design, response


def residual_analysis():
    plt.rcParams.update({'font.size': 22})
    plt.rcParams["figure.figsize"] = (14, 14)

    titles = ['Design', 'Unseen Points', 'Design with Added Scans',
              'Unseen Points with Added Scans']
    save_names = ['design', 'unseen', 'design_more_scans', 'unseen_more_scans']
    splits = ['train', 'all']
    samples = [0, 500]
    sig_irr = np.sqrt(S2_TOT - MATERN_SIG2)

    for i in range(4):
        design, response = resample_data(splits[i % 2], samples[i % 2])
        if i == 0:
            gp = fit_gp(design, response)

        pred = pred_gp(gp, design)
        diffs = pred-response
        ab_diff = abs(diffs)

        print('\n'+titles[i])
        for k in [1,2,3]:
            print(f'Fraction within {k} sd: {(ab_diff < k*sig_irr).mean()}')

        probplot(diffs, dist='norm', plot=plt)
        plt.title(titles[i]+' Residual QQ-Norm')
        plt.savefig(os.path.join('presentation_imgs', save_names[i]+'_qqnorm.png'), dpi=400)
        plt.cla(), plt.clf()
    

        plt.scatter(pred, diffs)
        plt.title(titles[i]+' Residuals vs Fitted Values')
        plt.xlabel('Accuracy')
        plt.savefig(os.path.join('presentation_imgs', save_names[i]+'_resid.png'), dpi=400)
        plt.cla(), plt.clf()
