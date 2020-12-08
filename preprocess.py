import tifffile
import skimage.transform
import os
import shutil
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.mixture import GaussianMixture

from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import IntVector


def resample_images(new_size, input_dir='tiff_images', output_dir='resampled_tiffs'):
    # new_size (int): new image sidelength
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    new_dir = os.path.join(output_dir, str(new_size))
    if os.path.isdir(new_dir):
        shutil.rmtree(new_dir)
    os.mkdir(new_dir)

    for tiff_file in os.listdir(input_dir):
        tiff_im = tifffile.imread(os.path.join(input_dir, tiff_file))
        tiff_rz = skimage.transform.resize(tiff_im, (new_size, new_size),
                                           anti_aliasing=True)
        tifffile.imwrite(os.path.join(new_dir, tiff_file), tiff_rz)


def pcareduce_images(reduc_dim, input_dir='tiff_images', output_dir='dimension_reduced'):
    # reduc_dim (int): new reduced dimensionality
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    new_dir = os.path.join(output_dir, str(reduc_dim))
    if os.path.isdir(new_dir):
        shutil.rmtree(new_dir)
    os.mkdir(new_dir)

    tiff_paths = [os.path.join(input_dir, tiff_file) for tiff_file in os.listdir(input_dir)]
    tiff_stack = tifffile.imread(tiff_paths)
    tiff_reduc = PCA(n_components=reduc_dim).fit_transform(normalize(
                                              tiff_stack.reshape(tiff_stack.shape[0], -1)))
    np.save(os.path.join(new_dir, 'all_data.npy'), tiff_reduc)


def gaussmix_images(n_components, input_dir='tiff_images', output_dir='gauss_mixtures'):
    # n_components (int): mixture components
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    new_dir = os.path.join(output_dir, str(n_components))
    if os.path.isdir(new_dir):
        shutil.rmtree(new_dir)
    os.mkdir(new_dir)

    dir1 = os.path.join(new_dir, 'comp_vector')
    dir2 = os.path.join(new_dir, 'comp_images')
    os.mkdir(dir1)
    os.mkdir(dir2)

    files = os.listdir(input_dir)

    n = len(files)
    arr1 = np.zeros((n, n_components))
    gauss_obj = GaussianMixture(n_components=n_components, n_init=3)

    for i in range(n):
        tiff_file = files[i]
        tiff_im = tifffile.imread(os.path.join(input_dir, tiff_file)).reshape(-1,1)
        gauss_obj.fit(tiff_im)

        mu = gauss_obj.means_
        arr1[i,:] = sorted(mu.flatten())

        tiff_cls = mu[gauss_obj.predict(tiff_im)].reshape(512, 512)
        tifffile.imwrite(os.path.join(dir2, tiff_file), tiff_cls)

    np.save(os.path.join(dir1, 'all_data.npy'), arr1)


def produce_all_datasets():
    for sz in [8, 16, 32, 64, 128]:
        resample_images(sz)

    for dim in [8, 16, 32, 64, 100]:
        pcareduce_images(dim)

    #for n_comp in range(5, 11):
    #    gaussmix_images(n_comp)

    design = gen_maxpro_pnts(500)

    np.save(os.path.join('experiment_data', 'design_points.npy'), design)


def to_design(unif_arr):
    unif_arr = np.array(unif_arr)
    unif_arr[:,0] = unif_arr[:,0]*45 + 5
    unif_arr[:,1] = unif_arr[:,1]*99 + 1
    unif_arr[:,2] = unif_arr[:,2]*990 + 10
    unif_arr[:,3] = np.floor(unif_arr[:,3]*2250 + 250)

    return unif_arr


def to_01(design_arr):
    design_arr = np.array(design_arr)
    design_arr[:,0] = (design_arr[:,0]-5)/45
    design_arr[:,1] = (design_arr[:,1]-1)/99
    design_arr[:,2] = (design_arr[:,2]-10)/990
    design_arr[:,3] = (design_arr[:,3]-250)/2250
    
    return design_arr


def gen_maxpro_pnts(N):
    rutils = importr('utils')
    rutils.chooseCRANmirror(ind=1)
    MaxPro = importr('MaxPro')

    cand = MaxPro.CandPoints(N, p_cont=3, l_disnum=IntVector([2251]))
    out_cand = MaxPro.MaxPro(cand)

    design_pnts = np.array(out_cand[0])

    return to_design(design_pnts)