# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 20:28:13 2019

@author: wtomjack
"""

from sklearn.svm import SVC #Support Vector Machine
import numpy as np
import time
from sklearn.model_selection import validation_curve, GridSearchCV
import matplotlib.pyplot as plt
class SupportVectorMachineExperiment(object):

    def __init__(self, verbose, trainX, trainY, kFolds, isHyper, isOne):
        self.verbose = verbose
        self.train(trainX, trainY, kFolds, isHyper, isOne)

    def train(self, trainX, trainY, kFolds, isHyper, isOne):
        startTime = time.time()

        if(isHyper):
            parameters = {"kernel":["linear", "rbf"], "C":np.arange(1,20)}
            self.svm = GridSearchCV(SVC(random_state=0), parameters, cv=kFolds)
        else:
            self.parameter_tuning(trainX, trainY, isOne)
            self.svm = SVC(random_state=0, C=1)

        self.svm.fit(trainX, trainY)
        endTime = time.time()
        self.trainingTime = endTime - startTime

        if(self.verbose):
            print("SVM Training Time: " + str(self.trainingTime))
            if(isHyper):
                print("SVM Params")
                print(self.svm.best_params_)
            print('')

    def parameter_tuning(self, trainX, trainY, isOne):
        kernelParameter = ["linear", "rbf"]
        cParameters = np.arange(1,20)

        trainScores1, testScores1 = validation_curve(SVC(random_state = 0), trainX, trainY, param_name="kernel", param_range=kernelParameter, cv=3)
        trainScores2, testScores2 = validation_curve(SVC(random_state = 0), trainX, trainY, param_name="C", param_range=cParameters, cv=3)

        plt.figure()
        plt.plot(kernelParameter, np.mean(trainScores1, axis=1), label="Training Score")
        plt.plot(kernelParameter, np.mean(testScores1, axis=1), label ="Cross Validation Score")
        plt.title("Validation Curve for Support Vector Machine")
        plt.xlabel("Kernel")
        plt.ylabel("Classification Score")
        plt.legend(loc="best")
        plt.grid()
        if(isOne):
            plt.savefig("plots/wine/Validation/SVM/Kernel.png")
        else:
            plt.savefig("plots/heart/Validation/SVM/Kernel.png")

        plt.figure()
        plt.plot(cParameters, np.mean(trainScores2, axis=1), label="Training Score")
        plt.plot(cParameters, np.mean(testScores2, axis=1), label ="Cross Validation Score")
        plt.title("Validation Curve for Support Vector Machine")
        plt.xlabel("C Penalty Patameter")
        plt.ylabel("Classification Score")
        plt.legend(loc="best")
        plt.grid()
        if(isOne):
            plt.savefig("plots/wine/Validation/SVM/Penalty.png")
        else:
            plt.savefig("plots/heart/Validation/SVM/Penalty.png")


    def query(self, testArr):
        startTime = time.time()
        classification = self.svm.predict(testArr)
        endTime = time.time()
        self.queryTime = endTime - startTime

        if(self.verbose) :
            #print("SVM Classification: " + str(classification))
            print("SVM Query Time: " + str(self.queryTime))

        return classification