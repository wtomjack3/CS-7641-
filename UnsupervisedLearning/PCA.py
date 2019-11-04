# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:48:43 2019

@author: wtomjack
"""
from sklearn.decomposition import PCA
import numpy as np

class PCAReduction(object):

    def __init__(self, trainDataX, n_components = 8, random_state = 10):
        self.pca = PCA(n_components, random_state = random_state, whiten=True)
        self.fit(trainDataX)

    def fit(self, trainData):
        self.xReduced = self.pca.fit_transform(trainData)
