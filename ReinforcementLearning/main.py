# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:38:24 2019

@author: wtomjack
"""

import gym
import pandas as pd
import numpy as np
from gym.envs.toy_text.frozen_lake import generate_random_map
import practice as mdp
import matplotlib.pyplot as plt
import mdptoolbox.example

def Run():
    NonGridWorldExperiment()
    GridWorldExperiment()
    SecondGridWorld()
    SecondNonGrid()

def run_episode(env, policy, gamma = .98, render = False):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done , _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        #print step_idx
        if done:
            print 'done'
            break
    return total_reward

def SecondNonGrid():
    prob, reward = mdptoolbox.example.forest(S = 5000)

    q = mdp.QLearning(prob, reward, .98, n_iter=50000)
    q.run()
    qdf = pd.DataFrame(q.run_stats)
    qdf.to_csv("C:/Users/wtomjack/.spyder/CS-7641-/ReinforcementLearning/QLEARNINGFORECT.csv")

    pi = mdp.PolicyIteration(prob, reward, 0.98, eval_type = 0, max_iter=5000)
    pi.run()
    ##print pi.policy
    pidf2 = pd.DataFrame(pi.run_stats)
    pidf2.to_csv("C:/Users/wtomjack/.spyder/CS-7641-/ReinforcementLearning/forest_pi0.csv")

    vi1 = mdp.ValueIteration(prob, reward, .98, max_iter=5000)
    vi1.run()
    #print vi.policy
    vidf = pd.DataFrame(vi1.run_stats)
    vidf.to_csv("C:/Users/wtomjack/.spyder/CS-7641-/ReinforcementLearning/forest_viXX.csv")
    PlotRewardsVsTimePiVi(pidf2, vidf)


def SecondGridWorld():
    #Create frozen lake env
    rand_map = generate_random_map(size=30, p = .8)
    env = gym.make("FrozenLake-v0")
    env.reset()

    nA, nS = env.nA, env.nS
    T = np.zeros([nA, nS, nS])
    R = np.zeros([nS, nA])

    for s in range(nS):
        for a in range(nA):
            transitions = env.P[s][a]
            for p_trans,next_s,rew,done in transitions:
                T[a,s,next_s] += p_trans
                R[s, a] = rew
        T[a,s,:]/=np.sum(T[a,s,:])

    q = mdp.QLearning(T, R, .98)
    q.run()
    qdf = pd.DataFrame(q.run_stats)
    qdf.to_csv("C:/Users/wtomjack/.spyder/CS-7641-/ReinforcementLearning/frozenQ.csv")

    pi = mdp.PolicyIteration(T, R, .98)
    pi.run()
    print len(pi.policy)

    vi = mdp.ValueIteration(T, R, .98)
    vi.run()
    print len(vi.policy)




#Cliff walking grid world problem
def GridWorldExperiment():
    #Create frozen lake env
    #rand_map = generate_random_map(size=4, p = .8)
    env = gym.make("FrozenLake-v0")
    env.reset()

    nA, nS = env.nA, env.nS
    T = np.zeros([nA, nS, nS])
    R = np.zeros([nS, nA])

    for s in range(nS):
        for a in range(nA):
            transitions = env.P[s][a]
            for p_trans,next_s,rew,done in transitions:
                T[a,s,next_s] += p_trans
                R[s, a] = rew
        T[a,s,:]/=np.sum(T[a,s,:])

    #value iteration
    viIterations = np.zeros(4)
    viTime = np.zeros(4)
    count = 0
    epsilonList = [0.05, 0.01, 0.005, 0.001]
    for i in range(len(epsilonList)):
        vi = mdp.ValueIteration(T, R, .98, epsilon= epsilonList[i], max_iter = 10000)
        vi.run()
        x = run_episode(env, vi.policy)
        #print vi.iter
        vidf = pd.DataFrame(vi.run_stats)
        vidf.to_csv("C:/Users/wtomjack/.spyder/CS-7641-/ReinforcementLearning/frozen_lake_vi.csv")
        viIterations[count] = len(vidf.index)
        viTime[count] = vidf.iloc[len(vidf.index)-1]["Time"]
        count +=1
    #PlotIterationsVsEpsilon("Plots/Grid/EpsilonVsIterationsVI.png", "Epsilon vs Iterations",  epsilonList, viIterations)
    #PlotTimeVsEpsilon("Plots/Grid/EpsilonVsTimeVI.png", "Epsilon vs Time", epsilonList, viTime)


    #Qlearning
    q = mdp.QLearning(T, R, 0.98)
    q.run()
    #print q.Q
    #x = run_episode(env, q.policy)
    qdf = pd.DataFrame(q.run_stats)
    qdf.to_csv("C:/Users/wtomjack/.spyder/CS-7641-/ReinforcementLearning/frozen_lake_q.csv")


    pi = mdp.PolicyIteration(T, R, .98, eval_type = 0)
    pi.run()
    y = run_episode(env, pi.policy)
    pidf = pd.DataFrame(pi.run_stats)
    print pi.policy
    pidf.to_csv("C:/Users/wtomjack/.spyder/CS-7641-/ReinforcementLearning/frozen_lake_pi2.csv")

    vi = mdp.ValueIteration(T, R, .98)
    vi.run()
    vidf = pd.DataFrame(vi.run_stats)
    print vi.policy
    vidf.to_csv("C:/Users/wtomjack/.spyder/CS-7641-/ReinforcementLearning/frozen_lake_VI1.csv")
    PlotVIVsPiTime(vidf, pidf)

    pi = mdp.PolicyIteration(T, R, .98, eval_type = 1)
    pi.run()
    y = run_episode(env, pi.policy)
    pidf1 = pd.DataFrame(pi.run_stats)


    PlotIterativeVsMatrixTime(pidf1, pidf)

def PlotVIVsPiTime(vi, pi):
    plt.plot(vi.index, vi["Time"], label="Value Iteration")
    plt.plot(pi.index, pi["Time"],  label="Policy Iteration")
    plt.legend(loc="best")
    plt.title("Policy Iteration vs Value Iteration Time")
    plt.xlabel("Iterations")
    plt.ylabel("Time")
    plt.grid()
    plt.savefig("Plots/Grid/PiVVi.png")


def NonGridWorldExperiment():
    #Establishes environment
    prob, reward = mdptoolbox.example.forest(S = 5000)


    vi = mdp.ValueIteration(prob, reward, .05, epsilon = .005, max_iter=5000)
    vi.run()
    #print vi.policy
    vidf = pd.DataFrame(vi.run_stats)
    vidf.to_csv("C:/Users/wtomjack/.spyder/CS-7641-/ReinforcementLearning/forest_vi.csv")
    PlotIterationsVsRewards("Plots/NonGrid/LowEpsilonHighDiscountVI.png", "Value Iteration Iterations vs Rewards", vidf['Reward'], vidf.index)


    vi = mdp.ValueIteration(prob, reward, .9, epsilon = .01, max_iter=5000)
    vi.run()
    #print vi.policy
    vidf = pd.DataFrame(vi.run_stats)
    vidf.to_csv("C:/Users/wtomjack/.spyder/CS-7641-/ReinforcementLearning/forest_vi.csv")
    PlotIterationsVsRewards("Plots/NonGrid/MedEpsilonHighDiscountVI.png", "Value Iteration Iterations vs Rewards", vidf['Reward'], vidf.index)

    vi = mdp.ValueIteration(prob, reward, .9, epsilon = .05, max_iter=5000)
    vi.run()
    #print vi.policy
    vidf = pd.DataFrame(vi.run_stats)
    vidf.to_csv("C:/Users/wtomjack/.spyder/CS-7641-/ReinforcementLearning/forest_vi.csv")
    PlotIterationsVsRewards("Plots/NonGrid/HighEpsilonLowDiscountVI.png", "Value Iteration Iterations vs Rewards", vidf['Reward'], vidf.index)


    discounts = [0.2, 0.4, 0.6, 0.8, 0.98]
    rewards = np.zeros(5)
    count = 0
    for i in discounts:
    #Policy Iteration
        pi = mdp.PolicyIteration(prob, reward, i,  max_iter=5000)
        pi.run()
        ##print pi.policy
        pidf1 = pd.DataFrame(pi.run_stats)
        pidf1.to_csv("C:/Users/wtomjack/.spyder/CS-7641-/ReinforcementLearning/forest_pi1.csv")
        rewards[count] =  pidf1.iloc[len(pidf1.index)-1]["Reward"]
        count +=1

    PlotDiscountVsReward(discounts, rewards)


    PlotIterativeVsMatrixTime(pidf1, pidf2)
    #Qlearning
    q = mdp.QLearning(prob, reward, 0.9, n_iter=10000)
    q.run()
    #print q.Q
    qdf = pd.DataFrame(q.run_stats)
    qdf.to_csv("C:/Users/wtomjack/.spyder/CS-7641-/ReinforcementLearning/forest_q.csv")
    PlotIterationsVsRewards("Plots/NonGrid/LowEpsilonHighDiscountQ.png", "Q Learning Iterations vs Rewards", qdf['Reward'], qdf.index)

def PlotRewardsVsTimePiVi(pi, vi):
    plt.plot(vi["Time"], vi["Reward"], "-o", label="Value Iteration")
    plt.plot(pi["Time"], pi["Reward"], "-o",  label="Policy Iteration")
    plt.legend(loc="best")
    plt.title("Time vs Rewards")
    plt.xlabel("Time")
    plt.ylabel("Rewards")
    plt.grid()
    plt.savefig("Plots/Grid/Pivii.png")

def PlotDiscountVsReward(discounts, rewards):
    plt.plot(discounts, rewards, "-o")
    plt.title("Algorithm Discount vs Rewards")
    plt.xlabel("Discount Value")
    plt.ylabel("Reward")
    plt.grid()
    plt.savefig("Plots/NonGrid/DiscountVsRewardPI.png")

def PlotIterationsVsRewards(filename, title, rewards, iterations):
    plt.figure()
    plt.title(title)
    plt.plot(iterations, rewards, "-")
    plt.xlabel("Iterations")
    plt.ylabel("Reward at Current Iteration")
    #plt.legend(loc="best")
    plt.grid()
    #plt.xlim(500, 5000)
    plt.savefig(filename)

def PlotIterationsVsEpsilon(filename, title, epsilon, iterations):
    plt.plot(epsilon, iterations)
    plt.title(title)
    plt.xlabel("Epsilon")
    plt.ylabel("Iterations")
    plt.grid()
    plt.savefig(filename)

def PlotIterativeVsMatrixTime(iterative, matrix):
    plt.plot(iterative.index, iterative["Time"], "-o", label="Iterative Evaluation Function")
    plt.plot(matrix.index, matrix["Time"], "-o", label="Matrix Evaluation Function")
    plt.title("Matrix Evaluation Time vs Iterative Evaluation Time")
    plt.xlabel("Iterations")
    plt.ylabel("Time")
    plt.grid()
    plt.savefig("Plots/NonGrid/TimeForIterativeVsMatrix.png")

def PlotTimeVsEpsilon(filename, title, epsilon, time):
    plt.figure()
    plt.title(title)
    plt.plot(epsilon, time, "-o")
    plt.ylim([0.0, 0.05])
    plt.xlabel("Epsilon")
    plt.ylabel("Time (Sec)")
    plt.grid()
    plt.savefig(filename)

if __name__ == "__main__":
    Run()
    print 'hello'

