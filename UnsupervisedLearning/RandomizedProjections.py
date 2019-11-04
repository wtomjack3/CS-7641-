# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:49:21 2019

@author: wtomjack
"""
from sklearn.random_projection import SparseRandomProjection
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt


class RandomizedProjectionReduction(object):

    def __init__(self, trainDataX, n_components = 8, random_state = 10, eps = 0.1):
        self.RandomizedProjection = SparseRandomProjection(n_components, random_state = random_state, eps = eps)
        self.fit(trainDataX)

    def fit(self, trainData):
        self.xReduced = self.RandomizedProjection.fit_transform(trainData)


    #Borrowed and adapted from SKlearn https://scikit-learn.org/stable/auto_examples/plot_johnson_lindenstrauss_bound.html#sphx-glr-auto-examples-plot-johnson-lindenstrauss-bound-pyhttps://scikit-learn.org/stable/auto_examples/plot_johnson_lindenstrauss_bound.html#sphx-glr-auto-examples-plot-johnson-lindenstrauss-bound-py
    def PairwiseDistribution(self, data):
        n_samples, n_features = data.shape
        print("Embedding %d samples with dim %d using various random projections"% (n_samples, n_features))

        n_components = 4
        dists = euclidean_distances(data, squared=True).ravel()

        # select only non-identical samples pairs
        nonzero = dists != 0
        dists = dists[nonzero]

        rp = SparseRandomProjection(n_components=n_components)
        projected_data = rp.fit_transform(data)
        projected_dists = euclidean_distances(
        projected_data, squared=True).ravel()[nonzero]
        plt.figure()
        plt.hexbin(dists, projected_dists, gridsize=100, cmap=plt.cm.PuBu)
        plt.xlim([0, 150])
        plt.ylim([0,150])
        plt.xlabel("Pairwise squared distances in original space")
        plt.ylabel("Pairwise squared distances in projected space")
        plt.title("Pairwise distances distribution for n_components= 4")
        cb = plt.colorbar()
        cb.set_label('Sample pairs counts')
        #cb.ax.set_yticklabels(['0','250', '500', '750', '1000', '1250'])
        plt.savefig("Plots/RandomProjection/pairwisedist2.png")



