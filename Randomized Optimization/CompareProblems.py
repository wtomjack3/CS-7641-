# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 14:48:22 2019

@author: wtomjack
"""
from mlrose import (random_hill_climb, simulated_annealing, genetic_alg, mimic, DiscreteOpt, ContinuousOpt, FourPeaks, Knapsack, OneMax)
import numpy as np

class CompareProblems():

    def __init__(self, stateCount):
        self.generate_problems(stateCount)

    def generate_problems(self, stateCount):
        #Four peaks
        threshold = .05
        self.fourPeaksProblem = self.four_peaks(threshold, stateCount)

        #Knapsack
        max_weight_percent = .6
        self.knapsackProblem = self.knapsack( max_weight_percent, stateCount)

        #Count Ones
        self.countOnesProblem = self.count_ones(stateCount)

    def four_peaks(self, threshold, stateCount):
        initial = FourPeaks(t_pct=threshold)
        state = self.generate_test_state(stateCount, "FP")
        problem = DiscreteOpt(length = stateCount, fitness_fn = initial, maximize = True)
        problem.set_state(state)

        return problem

    def knapsack(self, max_weight_percent, stateCount):
        state, weights, values = self.generate_test_state(stateCount, "KS")
        initial = Knapsack(weights, values, max_weight_percent)
        problem = DiscreteOpt(length = stateCount, fitness_fn = initial, maximize = True)

        problem.set_state(state)
        return problem

    def count_ones(self, stateCount):
        stateCount = stateCount
        initial = OneMax()
        state = self.generate_test_state(stateCount, "CO")
        problem = DiscreteOpt(length=stateCount, fitness_fn = initial, maximize = True)
        problem.set_state(state)
        return problem

    def generate_test_state(self, stateSize, testProblem):
        if(testProblem == "CO"):
            state = np.random.choice(2, stateSize)
            return state
        if(testProblem == "FP"):
            state = np.random.choice(2, stateSize)
            return state;
        if(testProblem == "KS"):
            state = np.random.choice(2, stateSize)

            weights = np.random.choice(np.arange(1,100), stateSize)
            values = np.random.choice(np.arange(1,100), stateSize)
            return state, weights, values

