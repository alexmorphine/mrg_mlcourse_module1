import argparse
import pickle
import numpy as np
from load_mnist import *
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


parser = argparse.ArgumentParser()
parser.add_argument("-x_train_dir", help="путь к файлу, в котором лежат рекорды обучающей выборки")
parser.add_argument("-y_train_dir", help="путь к файлу, в котором лежат метки обучающей выборки")
parser.add_argument("-model_output_dir", help="путь к файлу, в который скрипт сохраняет обученную модель")

args = parser.parse_args()

fname_img_path = args.x_train_dir
fname_lbl_path = args.y_train_dir

names = [os.path.join(fname_img_path, 'train-images-idx3-ubyte'), os.path.join(fname_lbl_path, 'train-labels-idx1-ubyte')]

model_output_dir = os.path.join(args.model_output_dir, 'model.pkl')

unzip(names)
labels, pics = read(dataset='training', fname_img_path=fname_img_path, fname_lbl_path=fname_lbl_path)
X_train, X_val, y_train, y_val = Stratified(pics, labels, random_state=42)
mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 0.00001
idx = std.nonzero()
X_train = (X_train - mean)/std


eig_vecs, X_train = PCA(X_train, mean)
y_train = OneHot(y_train)


def Softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)


def CrossEntropyLoss(y_h, y):
    log_likelihood = -np.log(y_h[0, y.argmax(axis=1)])
    return np.sum(log_likelihood)


def CrossEntropyLoss_grad(y_h, y, dz=1, lr=0.001):
    y_h[0, y.argmax(axis=1)] -= 1
    return y_h


def train(X, Y, lr=0.001, epoches=1000):
    np.random.seed(1)
    classes = Y.shape[1]
    input_shape = X.shape[1]
    input_len = X.shape[0]

    # Ксавьер
    W = np.random.normal(scale=0.1, size=(classes, input_shape))
    b = np.random.normal(scale=0.1, size=(classes))
    for iter in range(epoches):
        sh = list(range(input_len))  # больше рандома богу рандома
        np.random.shuffle(sh)
        for i in range(input_len):
            x = X[sh[i]].reshape(1, -1)
            y = Y[sh[i]].reshape(1, -1)
            y_h = Softmax(np.dot(x, W.T) + b)
            #loss = CrossEntropyLoss(y_h, y)
            y_h = CrossEntropyLoss_grad(y_h, y)
            dW = (x.T.dot(y_h)).T
            db = np.sum(y_h, axis=0)

            W -= lr * dW
            b -= lr * db

    return W, b

def save_model(W, b, mean, std, eig_vecs, model_output_dir='model.pkl'):
    to_pickle = {'W': W, 'b': b, 'mean': mean, 'std': std, 'eig_vecs': eig_vecs}
    with open(model_output_dir, 'wb') as f:
        pickle.dump(to_pickle, f)


def select_LR(X_val, y_val, X_train, y_train, mean, std, eig_vecs, epoches=10):
    best_acc = -1
    y_val = OneHot(y_val)
    X_val = (X_val - mean) / std
    X_val = X_val.dot(eig_vecs)

    for lr in [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]:
        W, b = train(X_train, y_train, epoches=epoches, lr=lr)
        y_pred = make_solution(X_val, W, b)
        acc = accuracy(y_pred, y_val)
        if acc > best_acc:
            best_acc = acc
            save_model(W, b, mean, std, eig_vecs)


select_LR(X_val, y_val, X_train, y_train, mean, std, eig_vecs)
W, b, mean, std, eig_vecs = load_model(model_output_dir)


y_pred = make_solution(X_train, W, b)
cl = classification_report(np.argmax(y_train, axis=1), y_pred)
print(cl)