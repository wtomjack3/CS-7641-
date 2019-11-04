# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 21:40:45 2019

@author: wtomjack
"""
from sklearn.cluster import KMeans
import pandas as pd
import time
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_samples, silhouette_score, completeness_score, adjusted_mutual_info_score, accuracy_score
import KClustering as kC
import ExpectationMaximization as em
import PCA as pca
import Plotting as plotting
import ICA as ica
import numpy as np
import RandomizedProjections as rp
import VarianceThreshold as vt


class UnsupervisedLearning(object):

    def __init__(self):
        self.pullData()
        self.plotter = plotting.Plotting()
        self.ExperimentOne()
        self.ExperimentTwo()
        self.ExperimentThree()
        self.ExperimentFour()

    def ExperimentTwo(self):
        #Wine Data
        self.winePCAData = pca.PCAReduction(self.wineTrainX)
        print("Wine Eigenvalues: ", self.winePCAData.pca.explained_variance_)
        print("Eigenvalue Variance" , self.winePCAData.pca.explained_variance_ratio_)
        self.wineICAData = ica.ICAReduction(self.wineTrainX)
        #print len(self.wineICAData.xReduced[1])
        #print len(self.wineICAData.xReduced)
        self.wineRandomizedProjectionData = rp.RandomizedProjectionReduction(self.wineTrainX)
        self.wineRandomizedProjectionData.PairwiseDistribution(self.wineTrainX)
        self.wineVarianceThresholdData = vt.VarianceThresholdReduction(self.wineTrainX)
        print 'xxxxxxx'
        for i in range(11):
            print np.var(self.wineTrainX[:, i])
        print 'yyyyyyyyyyyyyyy'
        for i in range(6):

            print np.var(self.wineVarianceThresholdData.xReduced[:,i])
        self.plotter.Plot_Data_Scatter("Plots/VarianceThreshold/wine.png", "Wine Data Points for First and Second Feature", self.wineTrainX[:, 0], self.wineTrainX[:, 1])
        self.plotter.Plot_Data_Scatter("Plots/VarianceThreshold/postReductionWine.png", "Wine Data Points for First and Second Feature", self.wineVarianceThresholdData.xReduced[:,0], self.wineVarianceThresholdData.xReduced[:, 1])

        #Heart Data
        self.heartPCAData = pca.PCAReduction(self.heartTrainX)
        print("heart Eigenvalues: ", self.heartPCAData.pca.explained_variance_)
        self.heartICAData = ica.ICAReduction(self.heartTrainX)
        self.heartRandomizedProjectionData = rp.RandomizedProjectionReduction(self.heartTrainX)
        self.heartVarianceThresholdData = vt.VarianceThresholdReduction(self.heartTrainX)
        print 'zzzzzzzzzzzzzzzzzz'
        print np.var(self.heartTrainX)
        print np.var(self.heartVarianceThresholdData.xReduced)
        for i in range(13):
            print np.var(self.heartTrainX[:, i])
        print 'yyyyyyyyyyyyyyy'
        for i in range(5):

            print np.var(self.heartVarianceThresholdData.xReduced[:,i])

        self.plotter.Plot_Data_Scatter("Plots/VarianceThreshold/heart.png", "Heart Data Points for First and Second Feature", self.heartTrainX[:, 0], self.heartTrainX[:, 1])
        self.plotter.Plot_Data_Scatter("Plots/VarianceThreshold/postReductionheart.png", "Heart Data Points for First and Second Feature", self.heartVarianceThresholdData.xReduced[:,0], self.heartVarianceThresholdData.xReduced[:, 1])

    def ExperimentThree(self):
        self.kClusteringExperimentPostTransformation(self.winePCAData.xReduced, self.heartPCAData.xReduced, "PCA")
        self.kClusteringExperimentPostTransformation(self.wineICAData.xReduced, self.heartICAData.xReduced, "ICA")
        self.kClusteringExperimentPostTransformation(self.wineRandomizedProjectionData.xReduced, self.heartRandomizedProjectionData.xReduced, "Randomized Projection")
        self.kClusteringExperimentPostTransformation(self.wineVarianceThresholdData.xReduced, self.heartVarianceThresholdData.xReduced, "Variance Threshold")

        self.ExpectationMaximizationPostTransformation(self.winePCAData.xReduced,self.heartPCAData.xReduced, "PCA")
        self.ExpectationMaximizationPostTransformation(self.wineICAData.xReduced,self.heartICAData.xReduced, "ICA")
        self.ExpectationMaximizationPostTransformation(self.wineRandomizedProjectionData.xReduced,self.heartRandomizedProjectionData.xReduced, "Randomized Projection")
        self.ExpectationMaximizationPostTransformation(self.wineVarianceThresholdData.xReduced,self.heartVarianceThresholdData.xReduced, "Variance Threshold")


    def ExperimentFour(self):
        self.NeuralNetworkApplication()
        classifier = MLPClassifier(hidden_layer_sizes=9, random_state=0, solver="lbfgs")
        clusterPCA = KMeans(n_clusters = 6).fit(self.winePCAData.xReduced)
        self.plotter.Plot_Learning_Curve(np.reshape(clusterPCA.labels_,(-1,1)), self.wineTrainY, classifier, \
                                    "Neural Network using Clustering", "Plots/NeuralNetworkClustering/PCA.png")

    def NeuralNetworkApplication(self):
        classifier = MLPClassifier(hidden_layer_sizes=9, random_state=0, solver="lbfgs")

        start = time.time()

        classifier.fit(self.wineVarianceThresholdData.xReduced, self.wineTrainY)

        end = time.time() - start
        print'time'
        print end
        self.plotter.Plot_Learning_Curve(self.winePCAData.xReduced, self.wineTrainY, classifier, \
                                    "Neural Network using PCA Transformation", "Plots/NeuralNetworkTransformation/PCA.png")


        self.plotter.Plot_Learning_Curve(self.wineICAData.xReduced, self.wineTrainY, classifier, \
                                    "Neural Network using ICA Transformation", "Plots/NeuralNetworkTransformation/ICA.png")

        self.plotter.Plot_Learning_Curve(self.wineRandomizedProjectionData.xReduced, self.wineTrainY, classifier, \
                                    "Neural Network using RandomizedProjection Transformation", "Plots/NeuralNetworkTransformation/RandomizedProjection.png")

        self.plotter.Plot_Learning_Curve(self.wineVarianceThresholdData.xReduced, self.wineTrainY, classifier, \
                                    "Neural Network using Variance Threshold Transformation", "Plots/NeuralNetworkTransformation/VarianceThreshold.png")



    def ExperimentOne(self):
        self.kClusteringExperiment()
        self.ExpectationMaximization()

    def kClusteringExperiment(self):
        i = 1
        wine_completeness = np.zeros(20)
        wine_silhouette = np.zeros(20)
        heart_silhouette = np.zeros(20)
        heart_completeness = np.zeros(20)
        #Wine Data Clustering
        while i < 21:
            i += 1
            c = i
            index = i-2
            cluster = kC.KClustering(clusters = c)
            cluster.fit(self.wineTrainX)
            prediction = cluster.predict(self.wineTrainX)
            silhouetteScore = silhouette_score(self.wineTrainX, cluster.kMeans.labels_)
            print( "Wine Data Silhouette Score for ", str(c), " clusters is ", silhouetteScore)
            fileName = "Plots/Clustering/wineClusterScatterPlot-" + str(i)+".png"
            #self.plotter.Plot_Clustering_Hist("Plots/Clustering/wineHistogram-" + str(i) +".png", " Wine Histogram", cluster.kMeans.labels_, i)
            self.plotter.Plot_Clustering_Scatter(fileName, "Wine Clustering", prediction, self.wineTrainX[:, 0], self.wineTrainX[:, 1])
            score = completeness_score(self.wineTrainY, prediction)
            wine_completeness[index] = score
            wine_silhouette[index] = silhouetteScore

        self.plotter.Plot_Silhouette("Plots/Clustering/wineSilhouette", "Wine Dataset Clustering Silhouette Score", wine_silhouette, 20)
        self.plotter.Plot_Cluster_Completeness("Plots/Clustering/wineCompleteness", "Wine Dataset Clustering Completeness", wine_completeness, 20)
        #Heart
        i = 1
        while i <21:
            i +=1
            c = i
            index = i-2
            cluster = kC.KClustering(clusters = c)
            cluster.fit(self.heartTrainX)
            silhouetteScore = silhouette_score(self.heartTrainX, cluster.kMeans.labels_)
            prediction = cluster.predict(self.heartTrainX)
            print ("Silhouette Score for " , str(c) , " clusters is " , silhouetteScore)
            fileName = "Plots/Clustering/heartClusterScatterPlot-" + str(i)+".png"
            #self.plotter.Plot_Clustering_Scatter(fileName, "Heart Clustering", prediction, self.heartTrainX[:, 2], self.heartTrainX[:, 1])
            score = completeness_score(self.heartTrainY, prediction)
            heart_completeness[index] = score
            heart_silhouette[index]= silhouetteScore
            #print silhouetteScore
        self.plotter.Plot_Silhouette("Plots/Clustering/heartSilhouette.png", "Heart Dataset Clustering Silhouette Score", heart_silhouette, 20)
        self.plotter.Plot_Cluster_Completeness("Plots/Clustering/heartCompleteness", "Heart Dataset Clustering Completeness", heart_completeness, 20)

    def ExpectationMaximization(self):
        i = 1
        wine_completeness = np.zeros(20)
        wine_silhouette = np.zeros(20)
        heart_silhouette = np.zeros(20)
        heart_completeness = np.zeros(20)
        #Wine Data Clustering
        while i < 21:
            i += 1
            c = i
            index = i-2
            cluster = em.ExpectationMaximization(n_components=i)
            cluster.fit(self.wineTrainX)
            prediction = cluster.predict(self.wineTrainX)
            #silhouetteScore = silhouette_score(self.wineTrainX, prediction)
            #print( "Silhouette Score for ", str(c), " clusters is ", silhouetteScore)
            fileName = "Plots/ExpectationMaximization/wineClusterScatterPlot-" + str(i)+".png"
            #self.plotter.Plot_Clustering_Scatter(fileName, "Wine Expectation Maximization", prediction, self.wineTrainX[:, 0], self.wineTrainX[:, 1])
            score = adjusted_mutual_info_score(self.wineTrainY, prediction)
            print("Wine NMI Score is " + str(score) + " when components number is " + str(i))

            wine_completeness[index] = cluster.EM.bic(self.wineTrainX)
            wine_silhouette[index]= score


        self.plotter.Plot_AMI("Plots/ExpectationMaximization/wineNormal.png", "Wine Dataset Gaussian Mixture AMI Score", wine_silhouette, 20)
        self.plotter.Plot_Cluster_BIC("Plots/ExpectationMaximization/wineBIC", "Wine Dataset EM BIC", wine_completeness, 20)
        #Heart
        i = 1
        while i <21:
            i +=1
            c = i
            index = i-2
            cluster = em.ExpectationMaximization(n_components=i)
            cluster.fit(self.heartTrainX)
            #silhouetteScore = silhouette_score(self.heartTrainX, prediction)
            prediction = cluster.predict(self.heartTrainX)
            #print ("Silhouette Score for " , str(c) , " clusters is " , silhouetteScore)
            fileName = "Plots/ExpectationMaximization/heartClusterScatterPlot-" + str(i)+".png"
            #self.plotter.Plot_Clustering_Scatter(fileName, "Heart Expectation Maximization", prediction, self.heartTrainX[:, 2], self.heartTrainX[:, 1])
            score = adjusted_mutual_info_score(self.heartTrainY, prediction)
            print("Heart NMI Score is " + str(score) + " when components number is " + str(i))
            heart_completeness[index] = cluster.EM.bic(self.heartTrainX)
            heart_silhouette[index]= score

            #print silhouetteScore
        self.plotter.Plot_Cluster_BIC("Plots/ExpectationMaximization/heartBIC", "Heart Dataset EM BIC", heart_completeness, 20)
        self.plotter.Plot_AMI("Plots/ExpectationMaximization/heartNormalization.png", "Heart Dataset Gaussian Mixture AMI Score", heart_silhouette, 20)


    def kClusteringExperimentPostTransformation(self, wineData, heartData, title):
        i = 1
        wine_completeness = np.zeros(20)
        wine_silhouette = np.zeros(20)
        heart_silhouette = np.zeros(20)
        heart_completeness = np.zeros(20)
        #Wine Data Clustering
        while i < 21:
            i += 1
            c = i
            index = i-2
            cluster = kC.KClustering(clusters = c)
            cluster.fit(wineData)
            prediction = cluster.predict(wineData)
            silhouetteScore = silhouette_score(wineData, cluster.kMeans.labels_)
            print( "Wine Data Silhouette Score for ", str(c), " clusters is ", silhouetteScore)
            fileName = "Plots/PostClustering/wineClusterScatterPlot-" + str(i)+ str(title) + ".png"
            #self.plotter.Plot_Clustering_Hist("Plots/PostClustering/wineHistogram-" + str(i) +title +".png", " Wine Histogram", cluster.kMeans.labels_, i)
            self.plotter.Plot_Clustering_Scatter(fileName, "Wine Clustering for " + str(title), prediction, wineData[:, 0], wineData[:, 1])
            score = completeness_score(self.wineTrainY, prediction)
            wine_completeness[index] = score
            wine_silhouette[index] = silhouetteScore

        self.plotter.Plot_Silhouette("Plots/PostClustering/wineSilhouette" + str(title), str(title) + "Wine Dataset Clustering Silhouette Score", wine_silhouette, 20)
        self.plotter.Plot_Cluster_Completeness("Plots/PostClustering/wineCompleteness" + str(title), str(title) +  "Wine Dataset Clustering Completeness", wine_completeness, 20)
        #Heart
        i = 1
        while i <21:
            i +=1
            c = i
            index = i-2
            cluster = kC.KClustering(clusters = c)
            cluster.fit(heartData)
            silhouetteScore = silhouette_score(heartData, cluster.kMeans.labels_)
            prediction = cluster.predict(heartData)
            print ("Silhouette Score for " , str(c) , " clusters is " , silhouetteScore)
            fileName = "Plots/PostClustering/heartClusterScatterPlot-" + str(i)+ str(title)+".png"
            self.plotter.Plot_Clustering_Scatter(fileName, str(title) + "Heart Clustering", prediction, heartData[:, 2], heartData[:, 1])
            score = completeness_score(self.heartTrainY, prediction)
            heart_completeness[index] = score
            heart_silhouette[index]= silhouetteScore
            #print silhouetteScore
        self.plotter.Plot_Silhouette("Plots/PostClustering/heartSilhouette" + str(title), str(title)+ "Heart Dataset Clustering Silhouette Score", heart_silhouette, 20)
        self.plotter.Plot_Cluster_Completeness("Plots/PostClustering/heartCompleteness" + str(title), str(title)+ "Heart Dataset Clustering Completeness", heart_completeness, 20)


    def ExpectationMaximizationPostTransformation(self, wineTrainX, heartTrainX, title):
        i = 1
        wine_completeness = np.zeros(20)
        wine_silhouette = np.zeros(20)
        heart_silhouette = np.zeros(20)
        heart_completeness = np.zeros(20)
        #Wine Data Clustering
        while i < 21:
            i += 1
            c = i
            index = i-2
            cluster = em.ExpectationMaximization(n_components=i)
            cluster.fit(wineTrainX)
            prediction = cluster.predict(wineTrainX)
            #silhouetteScore = silhouette_score(self.wineTrainX, prediction)
            #print( "Silhouette Score for ", str(c), " clusters is ", silhouetteScore)
            fileName = "Plots/PostExpectationMaximization/wineClusterScatterPlot-" + str(i)+ title+".png"
            self.plotter.Plot_Clustering_Scatter(fileName, title + "Wine Expectation Maximization", prediction, self.wineTrainX[:, 0], self.wineTrainX[:, 1])
            score = adjusted_mutual_info_score(self.wineTrainY, prediction)
            print("Wine NMI Score is " + str(score) + " when components number is " + str(i))

            wine_completeness[index] = cluster.EM.bic(wineTrainX)
            wine_silhouette[index]= score


        self.plotter.Plot_AMI("Plots/PostExpectationMaximization/wineNormal" + title, title + "Wine Dataset Gaussian Mixture AMI Score", wine_silhouette, 20)
        self.plotter.Plot_Cluster_BIC("Plots/PostExpectationMaximization/wineBIC" + title, title + "Wine Dataset EM BIC", wine_completeness, 20)
        #Heart
        i = 1
        while i <21:
            i +=1
            c = i
            index = i-2
            cluster = em.ExpectationMaximization(n_components=i)
            cluster.fit(heartTrainX)
            #silhouetteScore = silhouette_score(self.heartTrainX, prediction)
            prediction = cluster.predict(heartTrainX)
            #print ("Silhouette Score for " , str(c) , " clusters is " , silhouetteScore)
            fileName = "Plots/PostExpectationMaximization/heartClusterScatterPlot-" + str(i)+title+".png"
            self.plotter.Plot_Clustering_Scatter(fileName, title+ "Heart Expectation Maximization", prediction, self.heartTrainX[:, 2], self.heartTrainX[:, 1])
            score = adjusted_mutual_info_score(self.heartTrainY, prediction)
            print("Heart NMI Score is " + str(score) + " when components number is " + str(i))
            heart_completeness[index] = cluster.EM.bic(heartTrainX)
            heart_silhouette[index]= score

            #print silhouetteScore
        self.plotter.Plot_Cluster_BIC("Plots/PostExpectationMaximization/heartBIC" + title, title +"Heart Dataset EM BIC", heart_completeness, 20)
        self.plotter.Plot_AMI("Plots/PostExpectationMaximization/heartNormalization" + title, title+  "Heart Dataset Gaussian Mixture AMI Score", heart_silhouette, 20)

    def pullData(self):
            #Wine Training and Testing Data
            wineData =pd.read_csv("Data/winequality-red.csv", sep=";")
            wineX = wineData.iloc[:,:-1]
            wineX = preprocessing.scale(wineX)
            wineY = wineData.iloc[:,-1]
            wineY.loc[(wineY.ix[:] == 1) | (wineY.ix[:] == 2) | (wineY.ix[:] == 3) | (wineY.ix[:] == 4) | (wineY.ix[:] == 5)] = 0
            wineY.loc[(wineY.ix[:] == 6) | (wineY.ix[:] == 7) | (wineY.ix[:] == 8) | (wineY.ix[:] == 9)] = 1
            self.wineTrainX, self.wineTestX, self.wineTrainY, self.wineTestY = train_test_split(wineX, wineY, test_size=.25, random_state=0)
            #Adult Heart Data
            heartData = pd.read_csv("Data/processed.cleveland.data")
            heartX = heartData.iloc[:,:-1]
            heartX = preprocessing.scale(heartX)
            heartY = heartData.iloc[:,-1]
            heartY.loc[(heartY.ix[:] == 1) | (heartY.ix[:] == 2) | (heartY.ix[:] == 3) | (heartY.ix[:] == 4)] = 1


            self.heartTrainX, self.heartTestX, self.heartTrainY, self.heartTestY = train_test_split(heartX, heartY, test_size=.25, random_state=0)



if __name__ == "__main__":
    UnsupervisedLearning()
    print 'hello'