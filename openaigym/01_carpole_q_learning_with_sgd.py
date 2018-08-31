#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 12:09:21 2018

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


class SGDRegressor:
    def __init__(self, dim, lr=0.1):
        self.w = np.random.randn(dim) / np.sqrt(dim)
        self.lr = 0.1
    
    def update(self, X, y):
        self.w += self.lr*(y-X.dot(self.w)).dot(X)
    
    def predict(self, X):
        return X.dot(self.w)

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
    def __init__(self, env, feature_transformer):
        self.env = env
        self.feature_transformer = feature_transformer

        print("action space is ", env.action_space.n)
        #one SGD action
        self.models = [SGDRegressor(feature_transformer.dimensions) for _ in range(env.action_space.n)]

    def predict(self, s):
        X = self.feature_transformer.transform(np.atleast_2d(s))
        result = np.stack([m.predict(X) for m in self.models]).T
        return result
    
    def update(self, s, a, G):
        X = self.feature_transformer.transform(np.atleast_2d(s))
        self.models[a].update(X, [G])
        
    def sample_action(self, s, eps):
        if np.random.random_sample() > eps:
            return np.argmax(self.predict(s))
        else:
            return self.env.action_space.sample()

def play_one(env, model, eps, gamma):
    observation = env.reset()
    
    
    done=False
    steps = 0
    total_reward =0
    
    while not done:

        a = model.sample_action(observation, eps)
        prev_obs = observation
        observation, reward, done, info = env.step(a)

        
        if done and steps < 200:
            reward = -200
        G = reward + gamma*(np.max(model.predict(observation)))
        model.update(prev_obs, a, G)
        
        if not done:
            total_reward += reward
        steps +=1
    return total_reward

def main():
    gamma = 0.99
    env = gym.make('CartPole-v0')
    feat_t = FeatureTransformer()
    model = Model(env, feat_t)
    
    N = 500
    total_rewards = np.empty(N)
    for i in range(N):
        curr_eps = 1.0/np.sqrt(i+1)
        episode_reward = play_one(env, model,curr_eps, gamma)
        total_rewards[i] = episode_reward
        if i % 100 == 0:
            print("episode:", i, " last 100 r avg is: " , total_rewards[max(0,i-100):i+1].mean())
    print("finshed last 100 r avg is: " , total_rewards[-100:].mean())
    
    plt.plot(total_rewards)
    plt.title("Rewards against play")
    plt.show()
    
    plot_running_avg(total_rewards)
    
if __name__ == '__main__':
    main()