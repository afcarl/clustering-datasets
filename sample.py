#!/usr/bin/env python3

import os
import argparse

import numpy as np


def sample(dir, n):
    path = lambda file: os.path.join(dir, file)
    X, Y = np.load(path('X.npy')), np.load(path('Y.npy'))
    assert X.shape[0] == Y.shape[0]
    c = np.random.choice(X.shape[0], args.n, replace=False)
    X, Y = X[c], Y[c]
    dir = path(str(n))
    if not os.path.isdir(dir):
        os.makedirs(dir)
    np.save(os.path.join(dir, 'X.npy'), X)
    print(f"{os.path.join(dir, 'X.npy')} saved.")
    np.save(os.path.join(dir, 'Y.npy'), Y)
    print(f"{os.path.join(dir, 'Y.npy')} saved.")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Dataset Sampler')
    argparser.add_argument('datasets', metavar='<dir>', type=str, nargs='+', help='dir of the dataset')
    argparser.add_argument('-n', type=int, default=None, required=True)
    args = argparser.parse_args()
    for dataset in args.datasets:
        sample(dataset, n=args.n)
