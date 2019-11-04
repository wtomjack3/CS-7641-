# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:49:08 2019

@author: wtomjack
"""
from sklearn.decomposition import FastICA

class ICAReduction(object):

    def __init__(self, trainDataX, n_components = 10, random_state = 9):
        self.ica = FastICA(n_components, random_state = random_state, whiten=True)
        self.fit(trainDataX)

    def fit(self, trainData):
       self.xReduced = self.ica.fit_transform(trainData)