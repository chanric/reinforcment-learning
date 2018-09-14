#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 12:45:13 2018

@author: chari11
"""

import numpy as np
import gym
from carpole_train_with_sgd import FeatureTransformer, Model

"""
The main idea behind N steps is to base updates of the future N steps (with
a decaying weight gamma**n for each step).  It is about looking forward N steps to 
calc. When as N -> inf, we are basically doing monte carlo.  There are 2 phases to calc,
first is the in game calc.  The when the game is over and we need to estimate the N steps
where we don't have a reward value
"""
class SGDLazyInitRegressor:
    def __init__(self, dimensions=None, lr=1e-1):
        self._w = None if dimensions is None else np.random.randn(dimensions)/np.sqrt(dimensions)
        self._lr = lr
    
    def partial_fit(self, X, y):
        if self._w  is None:
            #X dim is 2nd arg. y length is first
            dim = X.shape[1]
            self._w = np.random.randn(dim)/np.sqrt(dim)
        self._w += self._lr*(y - X.dot(self._w)).dot(X)
        
    def predict(self, X):
        return X.dot(self._w)
    
class SGDLazyInitRegressorFactory:
    @staticmethod
    def create(dimensions, lr=0.1):
        return SGDLazyInitRegressor(lr=0.1)
    

    
def play_one(env, model, eps, gamma=0.9, n=5):
    observation = env.reset()
    states = []
    actions = []
    done = False
    rewards = []
    total_reward = 0
    i = 0
    curr_gamma = gamma**n
    multiplier = np.array([gamma]*n)**np.arange(n)
    while not done:
        
        action = model.sample_action(observation,eps)
        states.append(observation)
        actions.append(action)
        
        
        observation, reward, done, _ = env.step(action)
        #not sure if we need to punish for ending early.
        if done and i<200:
            reward = -20
        
        rewards.append(reward)
        if len(rewards) >= n:
            return_up_to_prediction = multiplier.dot(rewards[-n:])
            G = return_up_to_prediction + (curr_gamma)*np.max(model.predict(observation)[0])
            model.update(states[-n], actions[-n], G)

        if not done:
            total_reward += reward
        i += 1
    
    """
    if n == 1:
        rewards = []
        states = []
        actions = []
    else:
        rewards = rewards[-n+1:]
        states = states[-n+1:]
        actions = actions[-n+1:]
        """
    states = states[-n+1:]
    actions = actions[-n+1:]
    rewards = rewards[-n+1:]
    
    if i>=200:
        #aka win. all future rewards is zero
        while len(rewards) > 0:
            G = multiplier[:len(rewards)].dot(rewards)
            model.update(states[0], actions[0], G)
            rewards.pop(0)
            states.pop(0)
            actions.pop(0)
    else:
        while len(rewards) > 0:
            #we don't have the end state reward.. but we have to predict an reward
            #after in the future steps to calc win
            guess_rewards = rewards + [-20]*(n-len(rewards))
            G = multiplier.dot(guess_rewards)
            model.update(states[0], actions[0], G)
            rewards.pop(0)
            states.pop(0)
            actions.pop(0)
    
    return total_reward

if __name__ == '__main__':
    n = 5
    env = gym.make('CartPole-v0')
    f_transformer = FeatureTransformer()
    model = Model(env, f_transformer, SGDLazyInitRegressorFactory)
    gamma = 0.9
    
    N = 600
    total_rewards = np.empty(N)
    costs = np.empty(N)
    for i in range(N):
        curr_eps =  1.0/np.sqrt(i+1) 
        #0.1*(0.98**i)
        
        total_reward = play_one(env, model, curr_eps, gamma, n);
        total_rewards[i] = total_reward
        if i % 100 ==0:
            last_100_r = total_rewards[max(0,i-100):i].mean()
            #print("episode:", i, "total reward:", total_reward, "eps:", curr_eps)
            print("episode number: %s, last_100_reward_avg: %s, eps: %s" %( i, last_100_r, curr_eps))
    print("last 100 r is %s" %(total_rewards[-100:].mean()))
    print("total rewards:", total_rewards.sum())
    
    
        
        
        
        
        