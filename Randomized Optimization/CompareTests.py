# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 18:10:44 2019

@author: wtomjack
"""
from mlrose import (random_hill_climb, simulated_annealing, genetic_alg, mimic, GeomDecay)
import time
import matplotlib.pyplot as plt
import numpy as np

class CompareTests():
    def __init__(self):
        pass

    def setProblem(self, problemSpace):
        self.knapsack = problemSpace.knapsackProblem
        self.fourpeaks = problemSpace.fourPeaksProblem
        self.countones = problemSpace.countOnesProblem

    def plot(self, rhFitnessCurve, saFitnessCurve, gaFitnessCurve, mFitnessCurve, title, fileName):
        plt.figure()
        plt.title(title)
        mSize = mFitnessCurve.size
        gaSize = gaFitnessCurve.size
        saSize = saFitnessCurve.size
        rhSize = rhFitnessCurve.size
        plt.plot(np.arange(0, rhSize), rhFitnessCurve, "-", label="Random Hill Climbing")
        plt.plot(np.arange(0, saSize), saFitnessCurve, "-", label="Simulated Annealing")
        plt.plot(np.arange(0, gaSize), gaFitnessCurve, "-", label="Genetic Algorithm")
        plt.plot(np.arange(0, mSize), mFitnessCurve, "-", label="Mimic")
        plt.xlabel("Iterations")
        plt.ylabel("Fitness Value")
        plt.legend(loc="best")
        plt.grid()
        plt.xlim(0, 100)
        plt.savefig(fileName)

    def test(self):
        rhKS, rhFP, rhCO, rhKST, rhFPT, rhCOT, rhKSCurve, rhFPCurve, rhCOCurve = self.RandomHillClimbing()
        saKS, saFP, saCO, saKST, saFPT, saCOT, saKSCurve, saFPCurve, saCOCurve = self.SimulatedAnnealing()
        gaKS, gaFP, gaCO, gaKST, gaFPT, gaCOT, gaKSCurve, gaFPCurve, gaCOCurve = self.GeneticAlgorithm()
        mKS, mFP, mCO, mKST, mFPT, mCOT, mKSCurve, mFPCurve, mCOCurve = self.MIMIC()

        self.plot(rhKSCurve, saKSCurve, gaKSCurve, mKSCurve, "Knapsack over Iterations", "ksFitnessCurve.png")
        self.plot(rhFPCurve, saFPCurve, gaFPCurve, mFPCurve, "Four Peaks over Iterations", "fpFitnessCurve.png")
        self.plot(rhCOCurve, saCOCurve, gaCOCurve, mCOCurve, "Count Ones over Iterations", "coFitnessCurve.png")

        return rhKS, rhFP, rhCO, rhKST, rhFPT, rhCOT, rhKSCurve, rhFPCurve, rhCOCurve, saKS, saFP, saCO, saKST, saFPT, saCOT,  saKSCurve, saFPCurve, saCOCurve, \
        gaKS, gaFP, gaCO, gaKST, gaFPT, gaCOT, gaKSCurve, gaFPCurve, gaCOCurve, mKS, mFP, mCO, mKST, mFPT, mCOT, mKSCurve, mFPCurve, mCOCurve

    def RandomHillClimbing(self):
        start = time.time()
        KSbest_state, KSbest_fitness, KScurve = random_hill_climb(self.knapsack, curve=True)
        rhKST = time.time() - start

        start = time.time()
        FPbest_state, FPbest_fitness, FPcurve = random_hill_climb(self.fourpeaks, curve =True)
        rhFPT = time.time() - start

        start = time.time()
        CObest_state, CObest_fitness, COcurve = random_hill_climb(self.countones, curve = True)
        rhCOT = time.time() - start

        return KSbest_fitness, FPbest_fitness, CObest_fitness, rhKST, rhFPT, rhCOT, KScurve, FPcurve, COcurve

    def SimulatedAnnealing(self):
        start = time.time()
        KSbest_state, KSbest_fitness, KScurve= simulated_annealing(self.knapsack, curve = True)
        saKST = time.time() - start

        start = time.time()
        FPbest_state, FPbest_fitness, FPcurve = simulated_annealing(self.fourpeaks, curve = True)
        saFPT = time.time() - start

        start = time.time()
        CObest_state, CObest_fitness, COcurve = simulated_annealing(self.countones, schedule=GeomDecay(init_temp=.01, decay=.99), curve = True)
        saCOT = time.time() - start

        return KSbest_fitness, FPbest_fitness, CObest_fitness, saKST, saFPT, saCOT, KScurve, FPcurve, COcurve

    def GeneticAlgorithm(self):
        start = time.time()
        KSbest_state, KSbest_fitness, KScurve= genetic_alg(self.knapsack, curve= True)
        gaKST = time.time() - start

        start = time.time()
        FPbest_state, FPbest_fitness, FPcurve = genetic_alg(self.fourpeaks, curve= True, max_attempts = 20, pop_size = 400)
        gaFPT = time.time() - start

        start = time.time()
        CObest_state, CObest_fitness, COcurve = genetic_alg(self.countones, curve = True)
        gaCOT = time.time() - start

        return KSbest_fitness, FPbest_fitness, CObest_fitness, gaKST, gaFPT, gaCOT, KScurve, FPcurve, COcurve

    def MIMIC(self):
        start = time.time()
        KSbest_state, KSbest_fitness, KScurve=  mimic(self.knapsack, pop_size= 99, curve = True)
        mKST = time.time() - start

        start = time.time()
        FPbest_state, FPbest_fitness, FPcurve = mimic(self.fourpeaks, pop_size= 99, curve = True)
        mFPT = time.time() - start

        start = time.time()
        CObest_state, CObest_fitness, COcurve =  mimic(self.countones, pop_size= 99, curve = True)
        mCOT = time.time() - start

        return KSbest_fitness, FPbest_fitness, CObest_fitness, mKST, mFPT, mCOT, KScurve, FPcurve, COcurve



