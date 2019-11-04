# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 12:10:37 2019

@author: wtomjack
"""
from sklearn.feature_selection import VarianceThreshold

class VarianceThresholdReduction(object):

    def __init__(self, trainDataX):
        self.VarianceThreshold = VarianceThreshold(threshold = 1)
        self.fit(trainDataX)

    def fit(self, trainData):
        self.xReduced = self.VarianceThreshold.fit_transform(trainData)