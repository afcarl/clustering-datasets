#!/usr/bin/env python3

import os
import argparse

import numpy as np
from sklearn.neighbors import NearestNeighbors


def knn(dir, ks):
    path = lambda file: os.path.join(dir, file)
    X, Y = np.load(path('X.npy')), np.load(path('Y.npy'))
    assert X.shape[0] == Y.shape[0]
    ks = [k for k in ks if k < X.shape[0]]
    print('building %d-nn graph...' % max(ks))
    nn = NearestNeighbors(n_neighbors=max(ks), n_jobs=-1).fit(X).kneighbors(return_distance=False)
    for k in ks:
        np.save(path(f'{k}-nn.npy'), nn[:, :k])
        print(f"{path(f'{k}-nn.npy')} saved.")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Dataset k-NN Graph Generator')
    argparser.add_argument('datasets', metavar='<dir>', type=str, nargs='+', help='dir of the dataset')
    argparser.add_argument('--k', type=int, default=None)
    argparser.add_argument('--min', type=int, default=4)
    argparser.add_argument('--max', type=int, default=64)
    argparser.add_argument('--step', type=int, default=2)
    args = argparser.parse_args()
    ks = [args.k] if args.k is not None else np.arange(args.min, args.max+1, args.step, dtype=int)
    for dataset in args.datasets:
        knn(dataset, ks=ks)
