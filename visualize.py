#!/usr/bin/env python3

import os
import argparse

import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


def visualize(dir, show=True):
    path = lambda file: os.path.join(dir, file)
    X, Y = np.load(path('X.npy')), np.load(path('Y.npy'))
    assert X.shape[0] == Y.shape[0]

    if os.path.isfile(path('layout.npy')):
        print(f"{path('layout.npy')} exists. skip.")
        layout = np.load(path('layout.npy'))
    else:
        if X.shape[1] == 2:
            layout = X
        else:
            print('running t-sne...')
            layout = TSNE(n_components=2).fit_transform(X)

        np.save(path('layout.npy'), layout)
        print(f"{path('layout.npy')} saved.")
    plt.figure()
    plt.axis('off')
    pos = sorted([(layout[Y==label, 0], layout[Y==label, 1]) for label in np.unique(Y)], key=lambda s: (np.mean(s)))
    for x, y in pos:
        plt.scatter(x, y, s=4)
    plt.savefig(path('layout.png'))
    print(f"{path('layout.png')} saved.")
    if show:
        plt.show()
    plt.close()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Dataset Layout Generator')
    argparser.add_argument('datasets', metavar='<dir>', type=str, nargs='+', help='dir of the dataset')
    argparser.add_argument('--show', action='store_true')
    args = argparser.parse_args()
    for dataset in args.datasets:
        visualize(dataset, show=args.show)
