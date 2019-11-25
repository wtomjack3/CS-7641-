# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:16:22 2019

@author: wtomjack
"""

import gym
from gym.envs.toy_text.frozen_lake import generate_random_map

def Run():
    rand_map = generate_random_map(size=100, p = .8)
    env = gym.make("FrozenLake-v0", desc=rand_map)
    env.reset()
    env.render()