# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 18:27:01 2019

@author: wtomjack
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import validation_curve, train_test_split, learning_curve
from mlrose.neural import (gradient_descent, NetworkWeights, ContinuousOpt,
                           NeuralNetwork)
from mlrose.activation import identity, sigmoid, softmax
import time

class Experiment():

    def __init__(self):
        self.pullData()
        self.GeneticAlgorithmExperiment()
        self.RandomHillClimbExperiment()
        self.SimulatedAnnealingExperiment()
        self.Learning_Curve(self.wineTrainX, self.wineTrainY,  self.GAnetwork, "Learning Curve for NN using Genetic Algorithm Weight Optimization", 'GALearningCurve.png')
        self.Learning_Curve(self.wineTrainX, self.wineTrainY,  self.SAnetwork, "Learning Curve for NN using Simulated Annealing Weight Optimization", 'SALearningCurve.png')
        self.Learning_Curve(self.wineTrainX, self.wineTrainY,  self.RHCnetwork, "Learning Curve for NN using Random Hill Climb Weight Optimization", 'RHCLearningCurve.png')


    def test(self, actualY, classificationResults):
        accuracy = accuracy_score(actualY, classificationResults)
        #_, train_scores, test_scores = learning_curve(learner,xTrain, yTrain, train_sizes=train_sizes)
        return accuracy

    def GeneticAlgorithmExperiment(self):
        count = 1
        accuracyIterations = np.zeros([5, 4])
        while count <= 5:
            iterations = count * 500
            start = time.time()
            self.GAnetwork = NeuralNetwork(hidden_nodes=[2], activation='identity',
            algorithm='genetic_alg',
            bias=False, is_classifier=True,
            learning_rate=1, clip_max=1,
            max_attempts=100)

            weights = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
            #SET WEIGHTS AS A FUNCTION
            self.GAnetwork.fit(self.wineTrainX, self.wineTrainY, init_weights=weights)
            end = time.time() - start
            trainPrediction = self.GAnetwork.predict(self.wineTrainX)
            testPrediction = self.GAnetwork.predict(self.wineTestX)
            accuracyIterations[count-1, :] = [iterations, self.test(self.wineTrainY, trainPrediction), self.test(self.wineTestY, testPrediction), end]
            count = count + 1
            print count
        self.plotAccuracy(accuracyIterations, "Neural Network using Genetic Algorithm Weight Optimization", "Iterations", "Accuracy Percentage", "GAWeightOptimization.png")


    def RandomHillClimbExperiment(self):
        count = 1
        accuracyIterations = np.zeros([5, 4])
        while count <= 5:
            iterations = count * 500
            start = time.time()
            self.RHCnetwork = NeuralNetwork(hidden_nodes=[2], activation='relu',
                                            algorithm='random_hill_climb',
                                            bias=False, is_classifier=True,
                                            learning_rate=1, clip_max=1,
                                            max_attempts=100, max_iters = iterations)

            weights = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,1,1,1,1,1,1,1])
            self.RHCnetwork.fit(self.wineTrainX, self.wineTrainY, init_weights=weights)
            end = time.time()-start
            trainPrediction = self.RHCnetwork.predict(self.wineTrainX)
            testPrediction = self.RHCnetwork.predict(self.wineTestX)
            accuracyIterations[count-1, :] = [iterations, self.test(self.wineTrainY, trainPrediction), self.test(self.wineTestY, testPrediction), end]
            count = count + 1
            print count
        self.plotAccuracy(accuracyIterations, "Neural Network using Random Hill Climb Weight Optimization", "Iterations", "Accuracy Percentage", "RHCWeightOptimization.png")

    def SimulatedAnnealingExperiment(self):
        count = 1
        accuracyIterations = np.zeros([5, 4])
        while count <= 5:
            iterations = count * 500
            start = time.time()
            self.SAnetwork = NeuralNetwork(hidden_nodes=[2], activation='relu',
                                           algorithm='simulated_annealing',
                                           bias=False, is_classifier=True,
                                           learning_rate=1, clip_max=1,
                                           max_attempts=100, max_iters=iterations)
            weights = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
            self.SAnetwork.fit(self.wineTrainX, self.wineTrainY, init_weights=weights)
            end = time.time() - start
            #print "End time: " + str(end)
            trainPrediction = self.SAnetwork.predict(self.wineTrainX)
            testPrediction = self.SAnetwork.predict(self.wineTestX)
            accuracyIterations[count-1, :] = [iterations, self.test(self.wineTrainY, trainPrediction), self.test(self.wineTestY, testPrediction), end]
            count = count + 1
            print count
        self.plotAccuracy(accuracyIterations, "Neural Network using Simulated Annealing Weight Optimization", "Iterations", "Accuracy Percentage", "SAWeightOptimization.png")


    def plotAccuracy(self, data, title, xlabel, ylabel, filename):
        plt.figure()
        plt.title(title)
        plt.plot(data[:, 0], data[:, 1], "-o", label="Training Set Accuracy")
        plt.plot(data[:, 0], data[:, 2], "-o", label="Testing Set Accuracy")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc="best")
        plt.grid()
        plt.xlim(500, 5000)
        plt.savefig(filename)

    def Learning_Curve(self, xTrain, yTrain, learner, plotTitle, plotPath):
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


    def pullData(self):
        #Wine Training and Testing Data
        wineData =pd.read_csv("Data/winequality-red.csv", sep=";")
        wineX = wineData.iloc[:,:-1]
        wineX = preprocessing.scale(wineX)
        wineY = wineData.iloc[:,-1]
        wineY.loc[(wineY.ix[:] == 1) | (wineY.ix[:] == 2) | (wineY.ix[:] == 3) | (wineY.ix[:] == 4) | (wineY.ix[:] == 5)] = 0
        wineY.loc[(wineY.ix[:] == 6) | (wineY.ix[:] == 7) | (wineY.ix[:] == 8) | (wineY.ix[:] == 9)] = 1
        self.wineTrainX, self.wineTestX, self.wineTrainY, self.wineTestY = train_test_split(wineX, wineY, test_size=.25, random_state=0)


