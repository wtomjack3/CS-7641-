# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 20:25:28 2019

@author: wtomjack
"""

from sklearn.neural_network import MLPClassifier #Imports multilayer perceptron algorithm that trains using backpropagation
import numpy as np
import time
from sklearn.model_selection import validation_curve, GridSearchCV
import matplotlib.pyplot as plt

class NeuralNetworkExperiment(object):

    def __init__(self, verbose, trainX, trainY,kFolds, isHyperParameterized, isDataSet1):
        self.verbose = verbose
        self.train(trainX, trainY, kFolds, isHyperParameterized, isDataSet1)

    def train(self, trainX, trainY, kFolds, isHyper, isOne):

        startTime = time.time()
        if(isHyper):
            parameters = {"hidden_layer_sizes": np.arange(1,20),\
                      "alpha":[.0001, .001, .01, .1, 1, 10], "activation":['identity', 'logistic', 'tanh', 'relu']}
            self.ann = GridSearchCV(MLPClassifier(random_state=0), parameters, cv = kFolds)
        else:
            self.parameter_tuning(trainX, trainY, isOne)
            self.ann = MLPClassifier(random_state=0, hidden_layer_sizes=15, solver="lbfgs", alpha = .001)

        self.ann.fit(trainX, trainY)
        endTime = time.time()
        self.trainingTime = endTime- startTime

        if(self.verbose) :
            print("ANN Training Time: " + str(self.trainingTime))
            if(isHyper):
                print("ANN Params")
                print(self.ann.best_params_)
            print('')

    def parameter_tuning(self, trainX, trainY, isOne):
        hiddenLayerparameters = np.arange(1,20)
        AlphaParameters = [.0001, .001, .01, .1, .5, .9]

        trainScores1, testScores1 = validation_curve(MLPClassifier(random_state = 0), trainX, trainY, param_name="hidden_layer_sizes", param_range=hiddenLayerparameters, cv=3)
        trainScores2, testScores2 = validation_curve(MLPClassifier(random_state = 0), trainX, trainY, param_name="alpha", param_range=AlphaParameters, cv=3)

        plt.figure()
        plt.plot(hiddenLayerparameters, np.mean(trainScores1, axis=1), label="Training Score")
        plt.plot(hiddenLayerparameters, np.mean(testScores1, axis=1), label ="Cross Validation Score")
        plt.title("Validation Curve for Neural Network")
        plt.xlabel("Hidden Layer Size")
        plt.ylabel("Classification Score")
        plt.legend(loc="best")
        plt.grid()
        if(isOne):
            plt.savefig("plots/wine/Validation/ANN/HiddenLayer.png")
        else:
            plt.savefig("plots/heart/Validation/ANN/HiddenLayer.png")

        plt.figure()
        plt.plot(AlphaParameters, np.mean(trainScores2, axis=1), label="Training Score")
        plt.plot(AlphaParameters, np.mean(testScores2, axis=1), label ="Cross Validation Score")
        plt.title("Validation Curve for Neural Network")
        plt.xlabel("Regularization")
        plt.ylabel("Classification Score")
        plt.legend(loc="best")
        plt.grid()
        if(isOne):
            plt.savefig("plots/wine/Validation/ANN/Alpha.png")
        else:
            plt.savefig("plots/heart/Validation/ANN/Alpha.png")

    def query(self, testArr):
        startTime = time.time()
        classification = self.ann.predict(testArr)
        endTime = time.time()
        self.queryTime = endTime - startTime

        if(self.verbose) :
            #print("ANN Classification: " + str(classification))
            print("ANN Query Time: " + str(self.queryTime))

        return classification