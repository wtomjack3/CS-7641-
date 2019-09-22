# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 20:30:04 2019

@author: wtomjack
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn.model_selection import validation_curve, GridSearchCV

class KNearestNeighborExperiment(object):
    def __init__(self, verbose, trainX, trainY, kFolds, isHyper, isOne):
        self.verbose = verbose
        self.train(trainX, trainY, kFolds, isHyper, isOne)


    def train(self, trainX, trainY, kFolds, isHyper, isOne):
        #Run personal tests on n_neighbors, and p the minkowski metri, where p=1 is manhattan_distance, and p=2 is euclidean distance
        startTime = time.time()

        if(isHyper):
            parameters = {'n_neighbors': np.arange(1,40), "p":np.arange(1,2), "weights":['uniform','distance'], "algorithm":['ball_tree','kd_tree', 'brute', 'auto']}
            self.knn = GridSearchCV(KNeighborsClassifier(), parameters, cv=kFolds)
        else:
            self.parameter_tuning(trainX, trainY, isOne)
            self.knn = KNeighborsClassifier()

        self.knn.fit(trainX, trainY)
        endTime = time.time()
        self.trainingTime = endTime - startTime

        if(self.verbose):
            print("KNN Training Time: " + str(self.trainingTime))
            if(isHyper):
                print("KNN Params")
                print(self.knn.best_params_)
            print('')

    def parameter_tuning(self, trainX, trainY, isOne):
        neighborsParameters = np.arange(1,40)
        powerParameters = [1,2]

        trainScores1, testScores1 = validation_curve(KNeighborsClassifier(), trainX, trainY, param_name="n_neighbors", param_range=neighborsParameters, cv=3)
        trainScores2, testScores2 = validation_curve(KNeighborsClassifier(), trainX, trainY, param_name="p", param_range=powerParameters, cv=3)

        plt.figure()
        plt.plot(neighborsParameters, np.mean(trainScores1, axis=1), label="Training Score")
        plt.plot(neighborsParameters, np.mean(testScores1, axis=1), label ="Cross Validation Score")
        plt.title("Validation Curve for K-Nearest Neighbor")
        plt.xlabel("Number of Neighbors")
        plt.ylabel("Classification Score")
        plt.legend(loc="best")
        plt.grid()
        if(isOne):
            plt.savefig("plots/wine/Validation/KNN/Neighbors.png")
        else:
            plt.savefig("plots/heart/Validation/KNN/Neighbors.png")

        plt.figure()
        plt.plot(powerParameters, np.mean(trainScores2, axis=1), label="Training Score")
        plt.plot(powerParameters, np.mean(testScores2, axis=1), label ="Cross Validation Score")
        plt.title("Validation Curve for K-Nearest Neighbor")
        plt.xlabel("Euclidean vs Manhatten")
        plt.ylabel("Classification Score")
        plt.legend(loc="best")
        plt.grid()
        if(isOne):
            plt.savefig("plots/wine/Validation/KNN/Power.png")
        else:
            plt.savefig("plots/heart/Validation/KNN/Power.png")

    def query(self, testArr):
        startTime = time.time()
        classification = self.knn.predict(testArr)
        endTime = time.time()
        self.queryTime = endTime - startTime

        if(self.verbose) :
            #print("KNN Classification: " + str(classification))
            print("KNN Query Time: " + str(self.queryTime))

        return classification