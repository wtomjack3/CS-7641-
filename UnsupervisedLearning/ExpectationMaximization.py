# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:48:22 2019

@author: wtomjack
"""
from sklearn.mixture import GaussianMixture
class ExpectationMaximization(object):

    def __init__(self, n_components = 1, covariance_type = 'full', tol = .001):
        self.EM = GaussianMixture(n_components=n_components, tol = tol)

    def fit(self, xTrain):
        self.EM.fit(xTrain)

    def predict(self, y):
        return self.EM.predict(y)