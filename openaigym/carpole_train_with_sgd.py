#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 12:51:17 2018

@author: chari11
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from common_plots import plot_running_avg


class FeatureTransformer:
    def __init__(self):
        #normally use env.observation_space.sample() if all samples are likely
        observation_samples = np.random.random((20000,4))*2-1
        self.sc = StandardScaler()
        self.sc.fit(observation_samples)
        
        self.feature_union = FeatureUnion([
                ("rbf0", RBFSampler(gamma=0.05, n_components=1000)),
                ("rbf1", RBFSampler(gamma=0.1, n_components=1000)),
                ("rbf2", RBFSampler(gamma=0.5, n_components=1000)),
                ("rbf3", RBFSampler(gamma=1, n_components=1000))]
                )
        feature_examples = self.feature_union.fit_transform(self.sc.transform(observation_samples))

        self.dimensions = feature_examples.shape[1]
        print("feature example: ", feature_examples.shape)
    
    def transform(self, observations):
        X = self.sc.transform(observations)
        return self.feature_union.transform(X)

class Model:
    def __init__(self, env, feature_transformer, model_factory):
        self.env = env
        self.feature_transformer = feature_transformer

        print("action space is ", env.action_space.n)
        #one SGD action
        self.models = []
        for _ in range(env.action_space.n):
            model = model_factory.create(feature_transformer.dimensions) 
            model.partial_fit(feature_transformer.transform([env.reset()]), [0])
            self.models.append(model)

    def predict(self, s):
        X = self.feature_transformer.transform(np.atleast_2d(s))
        result = np.stack([m.predict(X) for m in self.models]).T
        return result
    
    def update(self, s, a, G):
        X = self.feature_transformer.transform(np.atleast_2d(s))
        self.models[a].partial_fit(X, [G])
        
    def sample_action(self, s, eps):
        if np.random.random_sample() > eps:
            return np.argmax(self.predict(s))
        else:
            return self.env.action_space.sample()

def play_one(env, model, eps, gamma=0.9):
    # Play games once based on model and return total reward
    # Updates G for model during plat
    #add env to arg?
    observation = env.reset()
    done = False
    total_reward = 0
    episode_num = 0
    while not done:
        action = model.sample_action(observation, eps)
        prev_o = observation
        observation, reward, done, _ = env.step(action)
        
        #punish for ending early
        if done and episode_num < 199:
            reward = -300
        
        total_reward += reward
        G = reward + gamma*np.max(model.predict(observation))
        model.update(prev_o, action, G)
        episode_num+=1
    return total_reward

    
    
        
def train_carpole_with_regressor(model_factory):
    env = gym.make('CartPole-v0')
    f_transformer = FeatureTransformer()
    model = Model(env, f_transformer, model_factory)
    gamma = 0.9
    
    N = 1000
    total_rewards = np.empty(N)
    for i in range(N):
        curr_eps = 1.0/np.sqrt(i+1)
        total_reward = play_one(env, model, curr_eps, gamma);
        total_rewards[i] = total_reward
        if i % 100 ==0:
            last_100_r = total_rewards[max(0,i-100):i].mean()
            #print("episode:", i, "total reward:", total_reward, "eps:", curr_eps)
            print("episode number: %s, last_100_reward_avg: %s, eps: %s" %( i, last_100_r, curr_eps))
    print("last 100 r is %s" %(total_rewards[-100:].mean()))
    print("total steps:", total_rewards.sum())
    
    plt.plot(total_rewards)
    plt.title("Rewards")
    plt.show()
    
    plot_running_avg(total_rewards)