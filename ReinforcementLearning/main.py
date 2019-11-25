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

def Run():
    #NonGridWorldExperiment()
    GridWorldExperiment()
    #print env.registry.all()

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
        print step_idx
        if done:
            print 'done'
            break
    return total_reward

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


    print T
    #value iteration
    vi = mdp.ValueIteration(T, R, 1, max_iter = 100)
    vi.run()
    x = run_episode(env, vi.policy)
    print vi.iter
    vidf = pd.DataFrame(vi.run_stats)
    vidf.to_csv("C:/Users/wtomjack/.spyder/CS-7641-/ReinforcementLearning/frozen_lake_vi.csv")

    #Qlearning
    q = mdp.QLearning(T, R, 0.98)
    q.run()
    #print q.Q
    #x = run_episode(env, q.policy)
    qdf = pd.DataFrame(q.run_stats)
    qdf.to_csv("C:/Users/wtomjack/.spyder/CS-7641-/ReinforcementLearning/frozen_lake_q.csv")


    pi = mdp.PolicyIteration(T, R, .98)
    pi.run()
    y = run_episode(env, pi.policy)
    pidf = pd.DataFrame(pi.run_stats)
    pidf.to_csv("C:/Users/wtomjack/.spyder/CS-7641-/ReinforcementLearning/frozen_lake_pi.csv")


def NonGridWorldExperiment():
    #Establishes environment
    prob, reward = mdptoolbox.example.forest(S = 2)

    print prob

    vi = mdp.ValueIteration(prob, reward, .9, epsilon = .005, max_iter=5000)
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


    #Policy Iteration
    pi = mdp.PolicyIteration(prob, reward, 0.9, eval_type = 1, max_iter=5000)
    pi.run()
    ##print pi.policy
    pidf = pd.DataFrame(pi.run_stats)
    pidf.to_csv("C:/Users/wtomjack/.spyder/CS-7641-/ReinforcementLearning/forest_pi.csv")
    PlotIterationsVsRewards("Plots/NonGrid/IterativeEvaluationPI.png", "Policy Iteration Iterations vs Rewards", pidf['Reward'], pidf.index)

    pi = mdp.PolicyIteration(prob, reward, 0.9, eval_type = 0, max_iter=5000)
    pi.run()
    ##print pi.policy
    pidf = pd.DataFrame(pi.run_stats)
    pidf.to_csv("C:/Users/wtomjack/.spyder/CS-7641-/ReinforcementLearning/forest_pi.csv")
    PlotIterationsVsRewards("Plots/NonGrid/MatrixEvaluationPI.png", "Policy Iteration Iterations vs Rewards", pidf['Reward'], pidf.index)



    #Qlearning
    q = mdp.QLearning(prob, reward, 0.9, n_iter=10000)
    q.run()
    #print q.Q
    qdf = pd.DataFrame(q.run_stats)
    qdf.to_csv("C:/Users/wtomjack/.spyder/CS-7641-/ReinforcementLearning/forest_q.csv")
    PlotIterationsVsRewards("Plots/NonGrid/LowEpsilonHighDiscountQ.png", "Q Learning Iterations vs Rewards", qdf['Reward'], qdf.index)


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

if __name__ == "__main__":
    Run()
    print 'hello'

