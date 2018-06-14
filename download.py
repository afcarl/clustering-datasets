#!/usr/bin/env python3

import os
import argparse

import numpy as np
import scipy.io as sio
from keras.utils.data_utils import get_file
from keras import datasets
from PIL import Image


def save(dir, X, Y):
    os.makedirs(dir) if not os.path.isdir(dir) else None
    print(f'{dir}: {{X:{X.shape}, Y:{Y.shape}, N:{len(np.unique(Y))}}}...', end=' ', flush=True)
    np.save(os.path.join(dir, 'X.npy'), X)
    np.save(os.path.join(dir, 'Y.npy'), Y)
    sio.savemat(os.path.join(dir, 'data.mat'), {'X': X, 'Y': Y})
    print('saved.')


def coil20():
    src = 'http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip'
    dir = os.path.join(os.path.dirname(get_file('coil-20-proc.zip', src, extract=True)), 'coil-20-proc')
    X  = []
    for i in range(1, 21):
        X += [np.array(Image.open(os.path.join(dir, f'obj{i}__{j}.png'))).flatten() for j in range(72)]
    X = np.array(X)
    Y = (np.ones([72, 20]) * np.arange(20)).T.flatten()
    save('coil20', X, Y)

def coil100():
    src = 'http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip'
    dir = os.path.join(os.path.dirname(get_file('coil-100.zip', src, extract=True)), 'coil-100')
    X = []
    for i in range(1, 101):
        X += [np.array(Image.open(os.path.join(dir, f'obj{i}__{j}.png'))).flatten() for j in range(0, 360, 5)]
    X = np.array(X)
    Y = (np.ones([72, 100]) * np.arange(100)).T.flatten()
    save('coil100', X, Y)

def mnist():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    save('mnist-test', np.reshape(x_test, [-1, 28*28]), y_test)
    X = np.reshape(np.concatenate([x_train, x_test]), [-1, 28 * 28])
    Y = np.concatenate([y_train, y_test])
    save('mnist', X, Y)

def pendigits():
    src  = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/'
    train = np.loadtxt(get_file('pendigits.tra', f'{src}pendigits.tra'), delimiter=',', dtype=int)
    test = np.loadtxt(get_file('pendigits.tes', f'{src}pendigits.tes'), delimiter=',', dtype=int)
    data = np.concatenate([train, test])
    X, Y = data[:, :-1], data[:, -1]
    save('pendigits', X, Y)

def usps():
    src = 'https://cs.nyu.edu/~roweis/data/usps_all.mat'
    X = np.transpose(sio.loadmat(get_file('usps_all.mat', src))['data'], axes=(2, 1, 0)).reshape([-1, 256])
    Y = (np.ones([1100, 10]) * np.arange(10)).T.flatten()
    save('usps', X, Y)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Dataset Downloader')
    argparser.add_argument('datasets', metavar='<name>', type=str, nargs='+', help='e.g., mnist')
    args = argparser.parse_args()
    for dataset in args.datasets:
        eval(dataset)()
