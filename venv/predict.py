import argparse
import pickle
import numpy as np
from load_mnist import *
import os
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser()
parser.add_argument("-x_test_dir", help="путь к файлу, в котором лежат рекорды тестовой выборки")
parser.add_argument("-y_test_dir", help="путь к файлу, в котором лежат метки тестовой выборки")
parser.add_argument("-model_input_dir", help="путь к файлу, из которого скрипт считывает обученную модель")

args = parser.parse_args()

fname_img_path = args.x_test_dir
print(fname_img_path)
fname_lbl_path = args.y_test_dir

names = [os.path.join(fname_img_path, 't10k-images-idx3-ubyte'), os.path.join(fname_lbl_path, 't10k-labels-idx1-ubyte')]
unzip(names)

model_input_dir = os.path.join(args.model_input_dir[0], 'model.pkl')


def load_model(model_input_dir):
    with open(model_input_dir, 'rb') as f:
        from_pickle = pickle.load(f)
        return from_pickle['W'], from_pickle['b'], from_pickle['mean'], from_pickle['std'], from_pickle['eig_vecs']

W, b, mean, std, eig_vecs = load_model(model_input_dir)


y, X = read(dataset='testing', fname_img_path=fname_img_path, fname_lbl_path=fname_lbl_path)

idx = std.nonzero()
X[:, idx] = (X[:, idx] - mean[idx])/std[idx]
X = X.dot(eig_vecs)
y = OneHot(y)

y_pred = make_solution(X, W, b)

cl = classification_report(np.argmax(y, axis=1), y_pred)
print(cl)