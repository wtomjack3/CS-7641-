# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 12:24:04 2019

@author: wtomjack
"""
import numpy as np
import matplotlib.pyplot as plt
import CompareProblems as probComp
import CompareTests as ct
import Experiment as ex
import time


def testAlgs():
    i = 1
    compareTests = ct.CompareTests()
    #State Size || RHC || SA || GA || M
    FourPeaksFitnessResults = np.zeros([5,5])
    KnapSackFitnessResults = np.zeros([5,5])
    CountOnesFitnessResults = np.zeros([5,5])

    FourPeaksComputeTime = np.zeros([5,5])
    KnapSackComputeTime = np.zeros([5,5])
    CountOnesComputeTime = np.zeros([5,5])
    while i < 6 :
        stateSize = i * 20
        compProb = probComp.CompareProblems(stateSize)
        compareTests.setProblem(compProb)
        rhKS, rhFP, rhCO, rhKST, rhFPT, rhCOT, rhKSCurve, rhFPCurve, rhCOCurve, saKS, saFP, saCO, saKST, saFPT, saCOT, saKSCurve, saFPCurve, saCOCurve, \
        gaKS, gaFP, gaCO, gaKST, gaFPT, gaCOT, gaKSCurve, gaFPCurve, gaCOCurve, mKS, mFP, mCO, mKST, mFPT, mCOT, mKSCurve, mFPCurve, mCOCurve = compareTests.test()
        FourPeaksFitnessResults[i-1, :] = [stateSize, rhFP, saFP, gaFP, mFP]
        KnapSackFitnessResults [i-1, :] = [stateSize, rhKS, saKS, gaKS, mKS]
        CountOnesFitnessResults[i-1, :] = [stateSize, rhCO, saCO, gaCO, mCO]
        FourPeaksComputeTime[i-1,:] = [stateSize, rhFPT, saFPT, gaFPT, mFPT]
        KnapSackComputeTime[i-1,:] = [stateSize, rhKST, saKST, gaKST, mKST]
        CountOnesComputeTime[i-1,:] = [stateSize, rhCOT, saCOT, gaCOT, mCOT]
        i = i + 1
        print i

    plotTestFitness("CountOnesFitnessTest.png", CountOnesFitnessResults, "N", "Fitness Value", "Count Ones Fitness")
    plotTestFitness("FourPeaksFitnessTest.png", FourPeaksFitnessResults, "N", "Fitness Value", "Four Peaks Fitness")
    plotTestFitness("KnapsackFitnessTest.png", KnapSackFitnessResults, "N", "Fitness Value", "Knapsack Fitness")

    plotTestFitness("CountOneComputeTime.png", CountOnesComputeTime, "N", "Time", "Count Ones Compute Time")
    plotTestFitness("FourPeaksComputeTime.png", FourPeaksComputeTime, "N", "Time", "Four Peaks Compute Time")
    plotTestFitness("KnapSackComputTime.png", KnapSackComputeTime, "N", "Time", "Knapsack Compute Time")

def plotTestFitness(fileName, data, labelX, labelY, title):
    plt.figure()
    plt.title(title)
    plt.plot(data[:, 0], data[:, 1], "-o", label="Random Hill Climbing")
    plt.plot(data[:, 0], data[:, 2], "-o", label="Simulated Annealing")
    plt.plot(data[:, 0], data[:, 3], "-o", label="Genetic Algorithm")
    plt.plot(data[:, 0], data[:, 4], "-o", label="Mimic")
    plt.xlabel(labelX)
    plt.ylabel(labelY)
    plt.legend(loc="best")
    plt.grid()
    plt.xlim(20, 100)
    plt.savefig(fileName)


if __name__ == "__main__":
    testAlgs()
    exp = ex.Experiment()
    print 'hello'


