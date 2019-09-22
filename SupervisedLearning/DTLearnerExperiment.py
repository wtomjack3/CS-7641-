# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 20:20:44 2019

@author: wtomjack
"""
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn import tree
from sklearn.model_selection import validation_curve, GridSearchCV

class DTLearnerExperiment(object):

    def __init__(self, verbose, trainingDataX, trainingDataY, kFolds, isHyper, isDataSet1):
        self.verbose = verbose
        self.train(trainingDataX, trainingDataY, kFolds, isHyper, isDataSet1)

    def train(self, trainX, trainY, kFolds, isHyper, isOne):
        ''' Need to manually tune 2 parameters in model and plot

            Use validation curve to help with this process'''
        startTime = time.time()
        if(isHyper):
            #Grid Search Returns the best possible parameter values
            parameters = {'max_depth': np.arange(1,50), 'min_samples_split': np.arange(2,50), 'criterion':['gini','entropy'], 'splitter':['best','random']}
            self.decisionTree = GridSearchCV(tree.DecisionTreeClassifier(random_state = 0), parameters, cv=kFolds)
        else:
            self.parameter_tuning(trainX, trainY , isOne)
            self.decisionTree = tree.DecisionTreeClassifier(random_state = 0, max_depth=12, min_samples_split = 2)

        self.decisionTree.fit(trainX, trainY)
        endTime = time.time()
        self.trainingTime = endTime - startTime

        if(self.verbose):
            print("DT Training Time: " + str(self.trainingTime))
            if(isHyper):
                print("DT Params")
                print(self.decisionTree.best_params_)
            print('')

    def parameter_tuning(self, trainX, trainY, isOne):
        depth_parameters = np.arange(1,31)
        min_samples_parameters = np.arange(2,30)

        trainScores1, testScores1 = validation_curve(tree.DecisionTreeClassifier(random_state = 0), trainX, trainY, param_name="max_depth", param_range=depth_parameters, cv=3)
        trainScores2, testScores2 = validation_curve(tree.DecisionTreeClassifier(random_state = 0), trainX, trainY, param_name="min_samples_split", param_range=min_samples_parameters, cv=3)

        plt.figure()
        plt.plot(depth_parameters, np.mean(trainScores1, axis=1), label="Decision Tree Training Score")
        plt.plot(depth_parameters, np.mean(testScores1, axis=1), label ="Cross Validation Score")
        plt.title("Validation Curve for Decision Tree")
        plt.xlabel("Tree Depth")
        plt.ylabel("Classification Score")
        plt.legend(loc="best")
        plt.grid()
        if(isOne):
            plt.savefig("plots/wine/Validation/DecisionTree/Depth.png")
        else:
            plt.savefig("plots/heart/Validation/DecisionTree/Depth.png")
        plt.clf()

        plt.figure()
        plt.plot(min_samples_parameters, np.mean(trainScores2, axis=1), label="Decision Tree Training Score")
        plt.plot(min_samples_parameters, np.mean(testScores2, axis=1), label ="Cross Validation Score")
        plt.title("Validation Curve for Decision Tree")
        plt.xlabel("Minimum Samples Split")
        plt.ylabel("Classification Score")
        plt.legend(loc="best")
        plt.grid()
        if(isOne):
            plt.savefig("plots/wine/Validation/DecisionTree/MinSamples.png")
        else:
            plt.savefig("plots/heart/Validation/DecisionTree/MinSamples.png")
        plt.clf()

    def query(self, testArr):
        startTime = time.time()
        classification = self.decisionTree.predict(testArr)
        endTime = time.time()
        self.queryTime = endTime - startTime

        if(self.verbose) :
            #print("DT Classification: " + str(classification))
            print("DT query time: " + str(self.queryTime))

        return classification