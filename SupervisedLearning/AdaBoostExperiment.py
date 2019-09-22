# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 14:28:44 2019

@author: wtomjack
"""
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.model_selection import validation_curve, GridSearchCV

class AdaBoostExperiment(object):

    def __init__(self, verbose, trainingDataX, trainingDataY, kFolds, isHyper, isOne):
        self.verbose = verbose
        self.train(trainingDataX, trainingDataY, kFolds, isHyper, isOne)


    def train(self, trainX, trainY, kFolds, isHyper, isOne):

        startTime = time.time()

        if(isHyper):
            abc = AdaBoostClassifier()
            parameters={'n_estimators':np.arange(1,300)}
            self.booster = GridSearchCV(abc, parameters, cv=kFolds)
        else:
            self.parameter_tuning(trainX, trainY, isOne)
            self.booster = AdaBoostClassifier(base_estimator = tree.DecisionTreeClassifier(random_state=0, max_features = "auto"), random_state=0, algorithm='SAMME')

        self.booster.fit(trainX, trainY)
        endTime = time.time()
        self.trainingTime = endTime - startTime

        if(self.verbose) :
            print("ADABoost Training Time: " + str(self.trainingTime))
            if(isHyper):
                print("ADA Params")
                print(self.booster.best_params_)
            print('')

    def parameter_tuning(self, trainX, trainY, isOne):
        estimatorsParameters = [50, 100, 150, 200, 250, 300]
        #LEARNING RATE PARAMETER WORKS AGAINST ESTIMATORS WRITE THAT SHIT UP IN PAPER
        learningRateParameters = [.001, .01, .1, 1]

        trainScores1, testScores1 = validation_curve(AdaBoostClassifier(), trainX, trainY, param_name="n_estimators", param_range=estimatorsParameters, cv=3)
        trainScores2, testScores2 = validation_curve(AdaBoostClassifier(), trainX, trainY, param_name="learning_rate", param_range=learningRateParameters, cv=3)

        plt.figure()
        plt.plot(estimatorsParameters, np.mean(trainScores1, axis=1), label="Training Score")
        plt.plot(estimatorsParameters, np.mean(testScores1, axis=1), label ="Cross Validation Score")
        plt.title("Validation Curve for ADA Boost with Decision Tree")
        plt.xlabel("Number of Estimators")
        plt.ylabel("Classification Score")
        plt.legend(loc="best")
        plt.grid()
        if(isOne):
            plt.savefig("plots/wine/Validation/ADA/Estimators.png")
        else:
            plt.savefig("plots/heart/Validation/ADA/Estimators.png")

        plt.figure()
        plt.plot(learningRateParameters, np.mean(trainScores2, axis=1), label="Training Score")
        plt.plot(learningRateParameters, np.mean(testScores2, axis=1), label ="Cross Validation Score")
        plt.title("Validation Curve for ADA Boost with Decision Tree")
        plt.xlabel("Learning Rate")
        plt.ylabel("Classification Score")
        plt.legend(loc="best")
        plt.grid()
        if(isOne):
            plt.savefig("plots/wine/Validation/ADA/LearningRate.png")
        else:
            plt.savefig("plots/heart/Validation/ADA/LearningRate.png")


    def query(self, testArr):
       startTime = time.time()
       classification = self.booster.predict(testArr)
       endTime = time.time()
       self.queryTime = endTime - startTime

       if(self.verbose) :
           #print("AdaBoost Classification: " + str(classification))
           print("AdaBoost Query Time: " + str(self.queryTime))

       return classification