# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 19:49:05 2019

@author: wtomjack
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import DTLearnerExperiment as DT
import NeuralNetworkExperiment as ANN
import SupportVectorMachineExperiment as SVM
import KNearestNeighborExperiment as KNN
import AdaBoostExperiment as ADA
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import validation_curve, train_test_split, learning_curve

class SupervisedLearningExperiment(object):

    def __init__(self, verbose, kFolds):
        self.verbose = verbose
        self.pullData()

        hyper = True
        notHyper = False

        #Wine Learners creation and training with Hyper Parameterizing
        self.wineDTLearnerHyper = DT.DTLearnerExperiment(verbose, self.wineTrainX, self.wineTrainY, kFolds, hyper,True)
        self.wineANNLearnerHyper = ANN.NeuralNetworkExperiment(verbose, self.wineTrainX, self.wineTrainY, kFolds, hyper, True)
        self.wineSVMLearnerHyper = SVM.SupportVectorMachineExperiment(verbose, self.wineTrainX, self.wineTrainY, kFolds, hyper, True)
        self.wineKNNLearnerHyper = KNN.KNearestNeighborExperiment(verbose, self.wineTrainX, self.wineTrainY, kFolds, hyper, True)
        self.wineADALearnerHyper = ADA.AdaBoostExperiment(verbose, self.wineTrainX, self.wineTrainY, kFolds, hyper, True)

        #Wine Learners creation with validation curve and default params PART 1
        self.wineDTLearnerNoHyperOne = DT.DTLearnerExperiment(verbose, self.wineTrainX, self.wineTrainY, kFolds, notHyper, True)
        self.wineANNLearnerNoHyperOne = ANN.NeuralNetworkExperiment(verbose, self.wineTrainX, self.wineTrainY, kFolds, notHyper, True)
        self.wineSVMLearnerNoHyperOne = SVM.SupportVectorMachineExperiment(verbose, self.wineTrainX, self.wineTrainY, kFolds, notHyper, True)
        self.wineKNNLearnerNoHyperOne = KNN.KNearestNeighborExperiment(verbose, self.wineTrainX, self.wineTrainY, kFolds, notHyper, True)
        self.wineADALearnerNoHyperOne = ADA.AdaBoostExperiment(verbose, self.wineTrainX, self.wineTrainY, kFolds, notHyper, True)


        #Income learners creation and training with Hyper Parameterizing
        self.heartDTLearnerHyper = DT.DTLearnerExperiment(verbose, self.heartTrainX, self.heartTrainY, kFolds, hyper, False)
        self.heartANNLearnerHyper = ANN.NeuralNetworkExperiment(verbose, self.heartTrainX, self.heartTrainY, kFolds, hyper, False)
        self.heartSVMLearnerHyper = SVM.SupportVectorMachineExperiment(verbose, self.heartTrainX, self.heartTrainY, kFolds, hyper, False)
        self.heartKNNLearnerHyper = KNN.KNearestNeighborExperiment(verbose, self.heartTrainX, self.heartTrainY, kFolds, hyper, False)
        self.heartADALearnerHyper = ADA.AdaBoostExperiment(verbose, self.heartTrainX, self.heartTrainY, kFolds, hyper, False)

        #Heart learners creation with validation curve and default params Part1
        self.heartDTLearnerNoHyperOne = DT.DTLearnerExperiment(verbose, self.heartTrainX, self.heartTrainY, kFolds, notHyper, False)
        self.heartANNLearnerNoHyperOne = ANN.NeuralNetworkExperiment(verbose, self.heartTrainX, self.heartTrainY, kFolds, notHyper, False)
        self.heartSVMLearnerNoHyperOne = SVM.SupportVectorMachineExperiment(verbose, self.heartTrainX, self.heartTrainY, kFolds, notHyper,False)
        self.heartKNNLearnerNoHyperOne = KNN.KNearestNeighborExperiment(verbose, self.heartTrainX, self.heartTrainY, kFolds, notHyper, False)
        self.heartADALearnerNoHyperOne = ADA.AdaBoostExperiment(verbose, self.heartTrainX, self.heartTrainY, kFolds, notHyper, False)


    def pullData(self):
        #Wine Training and Testing Data
        wineData =pd.read_csv("Data/winequality-red.csv", sep=";")
        wineX = wineData.iloc[:,:-1]
        wineX = preprocessing.scale(wineX)
        wineY = wineData.iloc[:,-1]
        self.wineTrainX, self.wineTestX, self.wineTrainY, self.wineTestY = train_test_split(wineX, wineY, test_size=.25, random_state=0)

        #Adult Heart Data
        heartData = pd.read_csv("Data/processed.cleveland.data")
        heartX = heartData.iloc[:,:-1]
        heartX = preprocessing.scale(heartX)
        heartY = heartData.iloc[:,-1]
        heartY.loc[(heartY.ix[:] == 1) | (heartY.ix[:] == 2) | (heartY.ix[:] == 3) | (heartY.ix[:] == 4)] = 1


        self.heartTrainX, self.heartTestX, self.heartTrainY, self.heartTestY = train_test_split(heartX, heartY, test_size=.25, random_state=0)
        pass


    def query(self):
        #Wine Data set
        wineSVMWithHyper = self.wineSVMLearnerHyper.query(self.wineTestX)
        wineKNNWithHyper = self.wineKNNLearnerHyper.query(self.wineTestX)
        wineANNWithHyper = self.wineANNLearnerHyper.query(self.wineTestX)
        wineDTWithHyper = self.wineDTLearnerHyper.query(self.wineTestX)
        wineADAWithHyper = self.wineADALearnerHyper.query(self.wineTestX)
        wHyperSvmAcc, wHyperKnnAcc, wHyperAnnAcc, wHyperDtAcc, wHyperAdaAcc =  \
        self.Measurements(wineSVMWithHyper, wineKNNWithHyper, wineANNWithHyper, wineDTWithHyper, wineADAWithHyper, self.wineTestY)

        self.Learning_Curve(self.wineTrainX, self.wineTrainY,  self.wineSVMLearnerHyper.svm.best_estimator_, "Learning Curve for SVM with Hyper Parameterization", 'plots/wine/LearningCurve/SVM/Hyper.png')
        self.Learning_Curve(self.wineTrainX, self.wineTrainY, self.wineKNNLearnerHyper.knn.best_estimator_, "Learning Curve for KNN with Hyper Parameterization", 'plots/wine/LearningCurve/KNN/Hyper.png')
        self.Learning_Curve(self.wineTrainX, self.wineTrainY, self.wineANNLearnerHyper.ann.best_estimator_, "Learning Curve for ANN with Hyper Parameterization", 'plots/wine/LearningCurve/ANN/Hyper.png')
        self.Learning_Curve(self.wineTrainX, self.wineTrainY, self.wineDTLearnerHyper.decisionTree.best_estimator_, "Learning Curve for DT with Hyper Parameterization", 'plots/wine/LearningCurve/DecisionTree/Hyper.png')
        self.Learning_Curve(self.wineTrainX, self.wineTrainY, self.wineADALearnerHyper.booster.best_estimator_, "Learning Curve for ADA with Hyper Parameterization", 'plots/wine/LearningCurve/ADA/Hyper.png')

        wineSVMNoHyperOne = self.wineSVMLearnerNoHyperOne.query(self.wineTestX)
        wineKNNNoHyperOne = self.wineKNNLearnerNoHyperOne.query(self.wineTestX)
        wineANNNoHyperOne = self.wineANNLearnerNoHyperOne.query(self.wineTestX)
        wineDTNoHyperOne = self.wineDTLearnerNoHyperOne.query(self.wineTestX)
        wineADANoHyperOne = self.wineADALearnerNoHyperOne.query(self.wineTestX)
        wNoHyperOneSvmAcc, wNoHyperOneKnnAcc, wNoHyperOneAnnAcc, wNoHyperOneDtAcc, wNoHyperOneAdaAcc = \
        self.Measurements(wineSVMNoHyperOne, wineKNNNoHyperOne, wineANNNoHyperOne, wineDTNoHyperOne,  wineADANoHyperOne, self.wineTestY)

        self.Learning_Curve(self.wineTrainX, self.wineTrainY,  self.wineSVMLearnerNoHyperOne.svm, "Learning Curve for SVM with Hyper Parameterization", 'plots/wine/LearningCurve/SVM/NoHyper.png')
        self.Learning_Curve(self.wineTrainX, self.wineTrainY, self.wineKNNLearnerNoHyperOne.knn, "Learning Curve for KNN with Hyper Parameterization", 'plots/wine/LearningCurve/KNN/NoHyper.png')
        self.Learning_Curve(self.wineTrainX, self.wineTrainY, self.wineANNLearnerNoHyperOne.ann, "Learning Curve for ANN with Hyper Parameterization", 'plots/wine/LearningCurve/ANN/NoHyper.png')
        self.Learning_Curve(self.wineTrainX, self.wineTrainY, self.wineDTLearnerNoHyperOne.decisionTree, "Learning Curve for DT with Hyper Parameterization", 'plots/wine/LearningCurve/DecisionTree/NoHyper.png')
        self.Learning_Curve(self.wineTrainX, self.wineTrainY, self.wineADALearnerNoHyperOne.booster, "Learning Curve for ADA with Hyper Parameterization", 'plots/wine/LearningCurve/ADA/NoHyper.png')

        if(self.verbose) :
            print("SVM Wine Accuracy: % " + str(wHyperSvmAcc * 100))
            #print("SVM Wine Acc !Hyper1: % " + str(wNoHyperOneSvmAcc *100))

            print("KNN Wine Accuracy: % " + str(wHyperKnnAcc * 100))
            #print("KNN Wine Acc !Hyper1: % " + str(wNoHyperOneKnnAcc *100))

            print("ANN Wine Accuracy: % " + str(wHyperAnnAcc * 100))
            #print("ANN Wine Acc !Hyper1: % " + str(wNoHyperOneAnnAcc *100))

            print("DT Wine Accuracy: % " + str(wHyperDtAcc * 100))
            #print("DT Wine Acc !Hyper1: % " + str(wNoHyperOneDtAcc *100))

            print("ADA Wine Accuracy: % " + str(wHyperAdaAcc * 100))
            #print("ADA Wine Acc !Hyper1: % " + str(wNoHyperOneAdaAcc *100))

        heartSVMWithHyper = self.heartSVMLearnerHyper.query(self.heartTestX)
        heartKNNWithHyper = self.heartKNNLearnerHyper.query(self.heartTestX)
        heartANNWithHyper = self.heartANNLearnerHyper.query(self.heartTestX)
        heartDTWithHyper = self.heartDTLearnerHyper.query(self.heartTestX)
        heartADAWithHyper = self.heartADALearnerHyper.query(self.heartTestX)
        hHyperSvmAcc, hHyperKnnAcc, hHyperAnnAcc, hHyperDtAcc, hHyperAdaAcc = \
        self.Measurements(heartSVMWithHyper, heartKNNWithHyper, heartANNWithHyper, heartDTWithHyper, heartADAWithHyper, self.heartTestY)

        self.Learning_Curve(self.heartTrainX, self.heartTrainY,  self.heartSVMLearnerHyper.svm.best_estimator_, "Learning Curve for SVM with Hyper Parameterization", 'plots/heart/LearningCurve/SVM/Hyper.png')
        self.Learning_Curve(self.heartTrainX, self.heartTrainY, self.heartKNNLearnerHyper.knn.best_estimator_, "Learning Curve for KNN with Hyper Parameterization", 'plots/heart/LearningCurve/KNN/Hyper.png')
        self.Learning_Curve(self.heartTrainX, self.heartTrainY, self.heartANNLearnerHyper.ann.best_estimator_, "Learning Curve for ANN with Hyper Parameterization", 'plots/heart/LearningCurve/ANN/Hyper.png')
        self.Learning_Curve(self.heartTrainX, self.heartTrainY, self.heartDTLearnerHyper.decisionTree.best_estimator_, "Learning Curve for DT with Hyper Parameterization", 'plots/heart/LearningCurve/DecisionTree/Hyper.png')
        self.Learning_Curve(self.heartTrainX, self.heartTrainY, self.heartADALearnerHyper.booster.best_estimator_, "Learning Curve for ADA with Hyper Parameterization", 'plots/heart/LearningCurve/ADA/Hyper.png')

        heartSVMNoHyperOne = self.heartSVMLearnerNoHyperOne.query(self.heartTestX)
        heartKNNNoHyperOne = self.heartKNNLearnerNoHyperOne.query(self.heartTestX)
        heartANNNoHyperOne = self.heartANNLearnerNoHyperOne.query(self.heartTestX)
        heartDTNoHyperOne = self.heartDTLearnerNoHyperOne.query(self.heartTestX)
        heartADANoHyperOne = self.heartADALearnerNoHyperOne.query(self.heartTestX)

        self.Learning_Curve(self.heartTrainX, self.heartTrainY,  self.heartSVMLearnerNoHyperOne.svm, "Learning Curve for SVM without Hyper Parameterization", 'plots/heart/LearningCurve/SVM/NoHyper.png')
        self.Learning_Curve(self.heartTrainX, self.heartTrainY, self.heartKNNLearnerNoHyperOne.knn, "Learning Curve for KNN without Hyper Parameterization", 'plots/heart/LearningCurve/KNN/NoHyper.png')
        self.Learning_Curve(self.heartTrainX, self.heartTrainY, self.heartANNLearnerNoHyperOne.ann, "Learning Curve for ANN without Hyper Parameterization", 'plots/heart/LearningCurve/ANN/NoHyper.png')
        self.Learning_Curve(self.heartTrainX, self.heartTrainY, self.heartDTLearnerNoHyperOne.decisionTree, "Learning Curve for DT without Hyper Parameterization", 'plots/heart/LearningCurve/DecisionTree/NoHyper.png')
        self.Learning_Curve(self.heartTrainX, self.heartTrainY, self.heartADALearnerNoHyperOne.booster, "Learning Curve for ADA without Hyper Parameterization", 'plots/heart/LearningCurve/ADA/NoHyper.png')


        if(self.verbose) :
            print("SVM Heart Accuracy: % " + str(hHyperSvmAcc * 100))

            print("KNN Heart Accuracy: % " + str(hHyperKnnAcc * 100))

            print("ANN Heart Accuracy: % " + str(hHyperAnnAcc * 100))

            print("DT Heart Accuracy: % " + str(hHyperDtAcc * 100))

            print("ADA Heart Accuracy: % " + str(hHyperAdaAcc * 100))

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
        pass


    def Measurements(self, svmClassification, knnClassification, annClassification, dtClassification, adaClassification, testData):
        #DISPLAY IN A TABLE
        svmAcc = accuracy_score(testData, svmClassification)
        knnAcc = accuracy_score(testData, knnClassification)
        annAcc = accuracy_score(testData, annClassification)
        dtAcc = accuracy_score(testData, dtClassification)
        adaAcc = accuracy_score(testData, adaClassification)

        return svmAcc, knnAcc, annAcc, dtAcc, adaAcc


if __name__ == "__main__":
    SLExperiment = SupervisedLearningExperiment(True, 3)
    SLExperiment.query()