#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 12:10:03 2018

@author: chari11
"""

import tensorflow as tf
import numpy as np
from common_plots import plot_running_avg, plot_x
import gym

class HiddenLayer:
    def __init__(self, input_dim, output_dim, activation_func=tf.nn.tanh):
        self._w = tf.Variable(tf.random_normal(shape=(input_dim, output_dim)))
        self._b = tf.Variable(np.zeros(output_dim).astype(np.float32))
        self._f = activation_func
    
    def forward(self, X):
        return self._f(tf.matmul(X, self._w) + self._b)
        
class PolicyModel:
    def __init__(self, input_dim, output_dim, hidden_layer_sizes):
        #output dim is num of acctions
        self._layers = []
        last_layer_size = input_dim
        for next_layer_size in hidden_layer_sizes:
            self._layers.append(HiddenLayer(last_layer_size, next_layer_size))
            last_layer_size = next_layer_size
        
        #final layer.  Note output_dim == num_actions
        self._layers.append(HiddenLayer(last_layer_size, output_dim, tf.nn.softmax))
        
        #input and targets
        self._X = tf.placeholder(tf.float32, shape=(None,input_dim), name='X')
        self._actions = tf.placeholder(tf.int32, shape=(None,), name='actions')
        self._advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')
        
        #construct full feed forward NN 
        Z = self._X
        for l in self._layers:
            Z = l.forward(Z)
        p_a_given_s = Z
        
        self._predict_op = p_a_given_s
        
        #Use one hot to zero prob of other actions
        select_probs = tf.log(
                tf.reduce_sum(
                        p_a_given_s * tf.one_hot(self._actions, output_dim),
                        reduction_indices=[1]))
        
        # negative because we want to MAX
        cost = -tf.reduce_sum(self._advantages * select_probs)
        
        self._train_op = tf.train.AdagradOptimizer(1e-1).minimize(cost)
    
    def partial_fit(self, X, actions, advantages):
        X = np.atleast_2d(X)
        actions = np.atleast_1d(actions)
        advantages = np.atleast_1d(advantages)
        self._session.run(
                self._train_op, 
                feed_dict={self._X: X,
                           self._advantages: advantages,
                           self._actions: actions})
    
    def predict(self, X):
        X = np.atleast_2d(X)
        return self._session.run(self._predict_op, feed_dict={self._X: X})[0]
    
    # note this doesn't need epsilon because the return type is an array
    def sample_action(self, X):
        p = self.predict(X)
        return np.random.choice(len(p), p=p)
    
    # can't put this in constructor. We must declare dependencies then init
    def set_session(self, session):
        self._session = session

class ValueModel:
    def __init__(self, input_dim, hidden_layer_sizes):
        self._layers = []
        #construct all layers up to output
        curr_in_size = input_dim
        for layer_node_num in hidden_layer_sizes:
            self._layers.append(HiddenLayer(curr_in_size, layer_node_num))
            curr_in_size = layer_node_num
        #construct output layer
        #note. no last output f
        self._layers.append(HiddenLayer(curr_in_size, 1, lambda x: x))
        
        self._X = tf.placeholder(tf.float32, shape=(None, input_dim), name='X')
        self._y = tf.placeholder(tf.float32, shape=(None,), name='y')
        
        Z = self._X
        for l in self._layers:
            Z = l.forward(Z)
        y_hat = tf.reshape(Z, [-1])
        self._predict_op = y_hat
        
        cost = tf.reduce_sum(tf.square(y_hat - self._y))
        
        self._train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(cost)
    
    def partial_fit(self, X, y):
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)
        self._session.run(
                self._train_op,
                feed_dict={ self._X: X, self._y: y})
        
    def predict(self, X):
        X = np.atleast_2d(X)
        return self._session.run(self._predict_op, 
                                 feed_dict={self._X: X})[0]
    
    # can't put this in constructor. We must declare dependencies then init
    def set_session(self, session):
        self._session = session

def play_one_td0(env, p_model, v_model, gamma):
    observation = env.reset()
    done = False
    total_reward = 0
    i = 0
    
    while not done:
        action = p_model.sample_action(observation)
        prev_o = observation
        observation, reward, done, _ = env.step(action)
        
        if done and i<200:
            reward = -150
        
        v_next = v_model.predict(observation)
        G = reward + gamma*np.max(v_next)
        advantage = G - v_model.predict(prev_o)
        p_model.partial_fit(prev_o, action, advantage)
        v_model.partial_fit(prev_o, G)
        
        if not done:
            total_reward += reward
        i += 1
    
    return total_reward
    
    
def play_one_mc(env, p_model, v_model, gamma):
    observation = env.reset()
    states = []
    actions = []
    rewards = []
    done = False
    total_rewards = 0
    reward = 0
    i = 0
    while not done:
        action = p_model.sample_action(observation)
        
        states.append(observation)
        actions.append(action)
        rewards.append(reward)
        
        observation, reward, done, _ = env.step(action)
        
        if not done:
            total_rewards += reward
        i += 1
    
    returns = []
    advantages = []
    G = 0
    for s, r in zip(reversed(states), reversed(rewards)):
        returns.append(G)
        advantages.append(G - v_model.predict(s))
        G = r + gamma*G
    
    returns.reverse()
    advantages.reverse()
    
    #update
    p_model.partial_fit(states, actions, advantages)
    v_model.partial_fit(states, returns)
    
    return total_rewards

def main_run(play_one=play_one_mc, policy_h_layers=[], value_h_layers=[10], gamma=0.99):
    env = gym.make('CartPole-v0')
    dim = env.observation_space.shape[0]
    
    p_model = PolicyModel(dim, env.action_space.n, policy_h_layers)
    v_model = ValueModel(dim, value_h_layers)
    
    #init session and create models
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)
    # With tensorflow, you have to set the session AFTER the init otherwise it
    # doesn't the dependencies
    p_model.set_session(session)
    v_model.set_session(session)
    
    N = 1000
    total_rewards = np.empty(N)
    for i in range(N):
        total_rewards[i] = play_one(env, p_model, v_model, gamma)
        if i % 100 ==0:
            last_100_r = total_rewards[max(0,i-100):i].mean()
            print("episode number: %s, last_100_reward_avg: %s" %( i, last_100_r))
    
    plot_x(total_rewards, 'rewards')
    plot_running_avg(total_rewards)
    

if __name__ == '__main__':
    # monte carlo
    #main_run(play_one=play_one_mc, policy_h_layers=[], value_h_layers=[10], gamma=0.99)
    
    #TD0
    main_run(play_one=play_one_td0, policy_h_layers=[4], value_h_layers=[25], gamma=0.90)

        
        
