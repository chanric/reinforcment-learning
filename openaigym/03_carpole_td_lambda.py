#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 12:21:34 2018

@author: chari11
"""

import numpy as np
from carpole_train_with_sgd import FeatureTransformer
import gym
from common_plots import plot_running_avg, plot_x

class RegressorModel:
    def __init__(self, dimensions):
        self._w = np.random.randn(dimensions)/np.sqrt(dimensions)
    
    def partial_fit(self, X_input, y, eligibility, lr=1e-2):
        self._w += lr*(y - X_input.dot(self._w))*eligibility
    
    def predict(self, X):
        X = np.array(X)
        return X.dot(self._w)

"""
TD lambda is a look bathwork method.  Note the addition of trhe eligibility vector
with decaying returns.
e.g. in update
        self._eligibilities *= gamma*lamdba_param
        self._eligibilities[a] += X[0]

"""
class LambdaModel:
    def __init__(self, env, feature_transformer):
        self._env = env
        self._feature_transformer = feature_transformer
        self._eligibilities = np.zeros((env.action_space.n, feature_transformer.dimensions))
        self._models = [RegressorModel(feature_transformer.dimensions) for _ in range(env.action_space.n) ]
    
    def predict(self, s):
        X = self._feature_transformer.transform(np.atleast_2d(s))
        return np.stack([m.predict(X) for m in self._models]).T
    
    def update(self, s, a, G, gamma, lamdba_param):
        X = self._feature_transformer.transform(np.atleast_2d(s))
        self._eligibilities *= gamma*lamdba_param
        self._eligibilities[a] += X[0]
        self._models[a].partial_fit(X[0], G, self._eligibilities[a])
    
    def sample_action(self, s, eps):
        if np.random.random() > eps:
            return np.argmax(self.predict(s))
        else:
            return self._env.action_space.sample()
    
    
def play_one(model, env, eps, gamma, lambda_param):
    observation = env.reset()
    done = False;
    total_reward = 0;
    i = 0
    while not done:
        action = model.sample_action(observation, eps)
        prev_o = observation
        observation, reward, done, _ = env.step(action)
        
        if done and i<200:
            reward = -200
        
        y_hat = model.predict(observation)
        assert(y_hat.shape == (1, env.action_space.n))
        G = reward + gamma*np.max(y_hat[0])
        model.update(prev_o, action, G, gamma, lambda_param)

        if not done:
            total_reward += reward
        i += 1
    return total_reward

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    feat_trans = FeatureTransformer()
    model = LambdaModel(env, feat_trans)
    gamma = 0.8
    lambda_param = 0.7
    
    N=500
    total_rewards = np.empty(N)
    for i in range(N):
        eps = 1.0/np.sqrt(i)
        ep_reward = play_one(model, env, eps, gamma, lambda_param)
        total_rewards[i] = ep_reward
        if i % 100 ==0:
            last_100_r = total_rewards[max(0,i-100):i].mean()
            print("episode number: %s, last_100_reward_avg: %s, eps: %s" %( i, last_100_r, eps))
    plot_x(total_rewards)
    plot_running_avg(total_rewards);
    
        
        
        
        