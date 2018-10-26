import os
import struct
import numpy as np
from sklearn.model_selection import train_test_split

import gzip
import shutil
import pickle
import os.path

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""


def read(dataset="training", fname_img_path=".", fname_lbl_path="."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(fname_img_path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(fname_lbl_path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(fname_img_path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(fname_lbl_path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), -1)

    return lbl, img


def select_PCA(eig_vals, eig_vecs, max_eig=165):
    selected_eig = []
    selected_eig_vecs = []
    for i, eig in enumerate(eig_vals):
#         if selected_eig.shape[1] == max_eig:
#             break
        if eig > 1:
            selected_eig.append(eig)
            selected_eig_vecs.append(eig_vecs[i])

    selected_eig = np.array(selected_eig)
    selected_eig_vecs = np.array(selected_eig_vecs).T
    return selected_eig, selected_eig_vecs


def PCA(X, mean):
    cov_mat = (X - mean).T.dot((X - mean)) / (X.shape[0] - 1)

    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[idx]
    _, eig_vecs = select_PCA(eig_vals, eig_vecs)

    return eig_vecs, X.dot(eig_vecs)

def make_solution(X, W, b):
    y_pred = np.dot(X, W.T) + b
    y_pred = np.argmax(y_pred, axis=1)
    return y_pred

def unzip(names):
    for filename in names:
        if not os.path.isfile(filename):
            with gzip.open(filename + '.gz', 'rb') as f_in:
                with open(filename, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            print('File {} already exists in the directory'.format(filename))

def OneHot(y):
    b = np.zeros((len(y), y.max() + 1))
    b[np.arange(len(y)), y] = 1
    return b

