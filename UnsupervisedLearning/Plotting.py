# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:25:50 2019

@author: wtomjack
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import completeness_score


class Plotting(object):

    def __init__(self):
        pass

    def Plot_Learning_Curve(self, xTrain, yTrain, learner, plotTitle, plotPath):
        train_sizes = np.linspace(0.1, 1.0, 5)
        _, train_scores, test_scores = learning_curve(learner,xTrain, yTrain, train_sizes=train_sizes)

        plt.figure()
        plt.plot(train_sizes, np.mean(train_scores, axis=1), '-o', label="Training Score")
        plt.plot(train_sizes, np.mean(test_scores, axis=1), '-o', label='Cross-validation score')
        plt.title(plotTitle)
        plt.xlabel('Number of training examples')
        plt.ylabel("Classification score")
        plt.legend(loc="best")
        plt.grid()
        plt.savefig(plotPath)

    def Plot_Clustering_Scatter(self, fileName, title, prediction, x, y):
        plt.figure()
        plt.scatter(x, y, c=prediction)
        plt.xlabel("First Feature Value")
        plt.ylabel("Second Feature Value")
        plt.title(title)
        plt.savefig(fileName)

    def Plot_Data_Scatter(self, fileName, title, x, y):
        plt.figure()
        plt.scatter(x, y)
        plt.xlabel("First Feature Value")
        plt.ylabel("Second Feature Value")
        plt.title(title)
        plt.savefig(fileName)

    def PlotICA(self, x, y):
        plt.figure()
        plt.scatter(x, y)
        plt.title("Distribution")

    def Plot_Clustering_Hist(self, fileName, title, labels, size):
        plt.figure()
        plt.hist(labels, bins=np.arange(0,size))
        plt.title(title)
        plt.savefig(fileName)
    #def Plot_DimensionalityReduction(self, fileName, title,):

    def Plot_Silhouette(self, fileName, title,  score, n_clusters):
        plt.figure()
        plt.plot(np.arange(1,21), score)
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Coefficient Value")
        plt.title(title)
        plt.savefig(fileName)
        plt.grid()
        plt.clf()

    def Plot_AMI(self, fileName, title,  score, n_clusters):
        plt.figure()
        plt.plot(np.arange(1,21), score)
        plt.xlabel("Number of Mixture Components")
        plt.ylabel("Adjusted Mutual Information Score")
        plt.title(title)
        plt.savefig(fileName)
        plt.grid()
        plt.clf()

    def Plot_Cluster_Completeness(self, fileName, title,  score, n_clusters):
        plt.figure()
        plt.plot(np.arange(1,21), score)
        plt.xlabel("Number of Clusters")
        plt.ylabel("Completeness Score")
        plt.title(title)
        plt.savefig(fileName)
        plt.grid()
        plt.clf()

    def Plot_Cluster_BIC(self, fileName, title,  score, n_clusters):
        plt.figure()
        plt.plot(np.arange(1,21), score)
        plt.xlabel("Number of Components")
        plt.ylabel("BIC")
        plt.title(title)
        plt.savefig(fileName)
        plt.grid()
        plt.clf()

