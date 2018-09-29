#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In policy gradient, we have 1 model for the policy AND 1 model for the value.  
The value model is simple regression.
The policy model use
    A range of samples with
        prob = self.predict(X)
        return np.random.choice(len(prob), p=prob)
    cost = -T.sum(advantages * T.log(p_a_given_s*T.))

Advantages. are cale with R going backwards with
        advantages.append(G-value_model.predict(s))
        G = r + gamma*G

@author: chari11
"""

import theano
import theano.tensor as T
import numpy as np
import gym
from common_plots import plot_running_avg, plot_x

class HiddenLayer:
    def __init__(self, input_layer_size, output_layer_size, activation_func = T.tanh):
        self._w = theano.shared(np.random.randn(input_layer_size, output_layer_size) * 
                                np.sqrt(2/input_layer_size))
        self._b = theano.shared(np.zeros(output_layer_size))
        self._af = activation_func
        self.params = [self._w, self._b]
        
    def forward(self, X):
        a =  X.dot(self._w) + self._b
        return self._af(a)
    
    

class PolicyModel:
    def __init__(self, dimensions, output_dim, hidden_layer_sizes, lr=1e-4):
        self._layers = []
        
        in_dim = dimensions
        for layer_size in hidden_layer_sizes:
            layer = HiddenLayer(in_dim, layer_size)
            self._layers.append(layer)
            in_dim = layer_size
        
        #do last layer
        final_layer = HiddenLayer(in_dim, output_dim, 
                                        activation_func = lambda x: x)
        self._layers.append(final_layer)
        
        params = []
        for l in self._layers:
            params += l.params
        
        #inputs
        X = T.matrix('X')
        #ivector for ints. Discret actions
        actions = T.ivector('actions')
        advantages = T.vector('advantages')
        
        Z = X
        for l in self._layers:
            Z = l.forward(Z)
        # soft max on action score
        p_a_given_s = T.nnet.softmax(Z)
        
        selected_probs = T.log(p_a_given_s[T.arange(actions.shape[0]), actions])
        cost = -T.sum(advantages * selected_probs)
        
        #create gradient decent on params
        grads = T.grad(cost, params)
        updates_with_gradient_decent = [(p, p - lr*g) for p, g in zip(params, grads)]
        
        self._train_op = theano.function(inputs=[X, actions, advantages],
                                         updates=updates_with_gradient_decent,
                                         allow_input_downcast=True)
        
        self._predict_op = theano.function(inputs=[X],
                                           outputs=p_a_given_s,
                                           allow_input_downcast=True)
    
    def partial_fit(self, X, actions, advantages):
        X = np.atleast_2d(X)
        actions = np.atleast_1d(actions)
        advantages = np.atleast_1d(advantages)
        self._train_op(X, actions, advantages)
        
    def predict(self, X):
        return self._predict_op(np.atleast_2d(X))[0]
    
    def sample_action(self, X):        
        #test for nan?
        prob = self.predict(X)
        return np.random.choice(len(prob), p=prob)
    
class ValueModel:
    def __init__(self, dimensions, hidden_layer_sizes, lr=1e-4):
        self._layers = []
        curr_in = dimensions
        for curr_out_size in hidden_layer_sizes:
            self._layers.append(HiddenLayer(curr_in, curr_out_size))
            curr_in = curr_out_size
        
        #do final layer
        #value only has 1 values
        self._layers.append(HiddenLayer(curr_in, 1))
        
        X = T.matrix('X')
        y = T.vector('y')
        
        params = []
        Z = X
        for layer in self._layers:
            Z = layer.forward(Z)
            params += layer.params
        y_hat = T.flatten(Z)
        
        
        self._predict_op = theano.function(inputs=[X],
                                           outputs=y_hat,
                                           allow_input_downcast=True)
        
        cost = T.sum((y_hat-y)**2)
        
        grads = T.grad(cost, params)
        update_params = [(p, p - lr*g) for p,g in zip(params, grads)]
        
        self._train_params = theano.function(inputs=[X,y],
                                updates=update_params,
                                allow_input_downcast=True)
        
    def partial_fit(self, X, y):
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)
        self._train_params(X,y)
    
    def predict(self, X):
        X = np.atleast_2d(X)
        return self._predict_op(X)[0]


def play_one_mc(env, policy_model, value_model, gamma):
    observation = env.reset()
    done = False;
    total_reward = 0
    
    states = []
    actions = []
    rewards = []
    i = 0
    reward = 0
    while not done:
        action = policy_model.sample_action(observation)

 
        actions.append(action)
        states.append(observation)
        rewards.append(reward)
        observation, reward, done, _ = env.step(action)        
        
        if done and i<200:
            #punish extra
            reward = -200
            

        if not done:
            total_reward += reward

        i+=1
    
    #do last iter
    action = policy_model.sample_action(observation)
    actions.append(action)
    states.append(observation)
    rewards.append(reward)
    
    returns = []
    advantages = []
    G = 0
    for s, r in zip(reversed(states), reversed(rewards)):
        returns.append(G)
        advantages.append(G-value_model.predict(s))
        G = r + gamma*G
    returns.reverse()
    advantages.reverse()
    
    policy_model.partial_fit(states[1:], actions[1:], advantages[1:])
    value_model.partial_fit(states, returns)
    
    return total_reward
    


def main(play_one=play_one_mc, policy_hidden_layers=[], value_hidden_layers=[10], gamma=0.99):
    env = gym.make('CartPole-v0')
    dim = env.observation_space.shape[0]
    K = env.action_space.n
    policy_model = PolicyModel(dim, K, policy_hidden_layers)
    value_model = ValueModel(dim, value_hidden_layers)

    
    N = 1000
    total_rewards = np.empty(N)
    for i in range(N):
        total_rewards[i] = play_one(env, policy_model, value_model, gamma)
        if i % 100 ==0:
            last_100_r = total_rewards[max(0,i-100):i].mean()
            print("episode number: %s, last_100_reward_avg: %s" %( i, last_100_r))
    
    plot_x(total_rewards, 'rewards')
    plot_running_avg(total_rewards)
    
if __name__ == '__main__':
    main()