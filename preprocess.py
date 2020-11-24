import tifffile
import skimage.transform
import os
import shutil
import numpy as np

from sklearn.decomposition import PCA


def resample_images(new_size, input_dir='tiff_images', output_dir='resampled_tiffs'):
    # Both input_dir and output_dir must exist before calling this function
    # new_size (int): new image sidelength

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
    # Both input_dir and output_dir must exist before calling this function
    # reduc_dim (int): new reduced dimensionality

    new_dir = os.path.join(output_dir, str(reduc_dim))
    if os.path.isdir(new_dir):
        shutil.rmtree(new_dir)
    os.mkdir(new_dir)

    tiff_paths = [os.path.join(input_dir, tiff_file) for tiff_file in os.listdir(input_dir)]
    tiff_stack = tifffile.imread(tiff_paths)
    tiff_reduc = PCA(n_components=reduc_dim).fit_transform(normalize(
                                              tiff_stack.reshape(tiff_stack.shape[0], -1)))
    np.save(os.path.join(new_dir, 'all_data.npy'), tiff_reduc)
