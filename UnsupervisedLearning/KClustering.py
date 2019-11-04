# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:39:58 2019

@author: wtomjack
"""
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

class KClustering(object):

    def __init__(self, clusters = 8, init =10, alg = "auto"):
        self.kMeans = KMeans(n_clusters=clusters, n_init= init, algorithm = alg, random_state = 0)

    def fit(self, trainX):
        self.kMeans.fit(trainX)

    def predict(self, y):
        return self.kMeans.predict(y)
