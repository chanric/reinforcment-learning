#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Note
Notes:
 - We have 2 copies of the same DQN.  We keep the params the of one the same and use
it to predict all the actions when training the other.  After a certain copy size,
we copy the training model to the prediction model while playing game
 - We don't train unless we have a certain miniumn number of experiences,
and we train based of a max number of experiences.
 - When we do train, we train based a randomly selected number of samples from 
the min max expereiences above

"""

import numpy as np
import tensorflow as tf
import gym
from common_plots import plot_running_avg, plot_x

class HiddenLayer:
    def __init__(self, input_dim, output_dim, act_func=tf.nn.tanh, use_bias=True):
        self._w = tf.Variable(tf.random_normal(shape=(input_dim, output_dim)))
        self.params = [self._w]
        self._is_bias_enabled = use_bias
        if use_bias:
            self._b = tf.Variable(np.zeros(output_dim).astype(np.float32))
            self.params.append(self._b)
        self._af = act_func
    
    def forward(self, X):
        if self._is_bias_enabled:
            a = tf.matmul(X, self._w) + self._b
        else:
            a = tf.matmul(X, self._w)
        return self._af(a)
    
class DeepQNetwork:
    def __init__(self, input_dim, output_dim, hidden_layer_sizes, gamma, scope,
                 max_experiences=10000, min_experiences=100, batch_size=32):
        self._output_dim = output_dim
        self._gamma = gamma
        self._max_experiences = max_experiences
        self._min_experiences = min_experiences
        self._batch_size = batch_size
        self._exp = []
        self.scope = scope
        
        with tf.variable_scope(scope):
            self._layers = []
            next_input_dim = input_dim
            for next_output_dim in hidden_layer_sizes:
                self._layers.append(HiddenLayer(next_input_dim, next_output_dim))
                next_input_dim =next_output_dim
            
            #last layer is 1-1
            last_layer = HiddenLayer(next_input_dim, output_dim, act_func=lambda x : x)
            self._layers.append(last_layer)
            
            self.params =[]
            for l in self._layers:
                self.params += l.params
            
            self._X = tf.placeholder(tf.float32, shape=(None, input_dim), name='X')
            self._G = tf.placeholder(tf.float32, shape=(None,), name='G')
            self._actions = tf.placeholder(tf.int32, shape=(None,), name='actions')
            
            Z = self._X
            for l in self._layers:
                Z = l.forward(Z)
            y_hat = Z
            self._predict_op = y_hat
            
            selected_action_values = tf.reduce_sum(
                    y_hat * tf.one_hot(self._actions, output_dim),
                    reduction_indices=[1])
            
            self._cost = tf.reduce_sum(tf.square(self._G - selected_action_values))
            self._train_op = tf.train.AdamOptimizer(1e-2).minimize(self._cost)
            #self._train_op = tf.train.AdagradOptimizer(1e-2).minimize(self._cost)
    
    def set_session(self, session):
        self._session = session
        
    
    def load_from(self):
        #rather than do a liv copy. we can presetup the copy
        self._session.run(self._copy_from_ops)
    
    def setup_copy_from(self, other):
        copy_ops = []
        for m, o in zip(self.params, other.params):
            op = m.assign(o)
            copy_ops.append(op)
        self._copy_from_ops = copy_ops
        
        
    def predict(self, X):
        X = np.atleast_2d(X)
        return self._session.run(self._predict_op,
                                 feed_dict={self._X: X})
        
    def train(self, target_network):
        if len(self._exp) < self._min_experiences:
            return
        
        indexes = np.random.choice(len(self._exp), size=self._batch_size, replace=False)
        states = [self._exp[i]['s'] for i in indexes]
        actions = [self._exp[i]['a'] for i in indexes]
        rewards = [self._exp[i]['r'] for i in indexes]
        next_states = [self._exp[i]['s_next'] for i in indexes]
        dones = [self._exp[i]['d'] for i in indexes]
        next_Q = np.max(target_network.predict(next_states), axis=1)
        targets = [r + self._gamma*next_q if not done else r 
                   for r, next_q, done in zip(rewards, next_Q, dones)]
        
        self._session.run(
                self._train_op,
                feed_dict={
                        self._X: states,
                        self._G: targets,
                        self._actions: actions})
    
    def add_experience(self, s, a, r, s_next, done):
        if len(self._exp) > self._max_experiences:
            self._exp.pop(0)
        
        self._exp.append({'s':s, 
                          'a': a, 
                          'r': r, 
                          's_next':s_next, 
                          'd': done})
    
    def sample_action(self, X, eps):
        if np.random.random() < eps:
            return np.random.choice(self._output_dim)
        else:
            return np.argmax(self.predict(X)[0])

def play_one(env, model, train_model, eps, copy_period):
    observation = env.reset()
    done = False
    total_reward =0
    i = 0
    
    while not done:
        action = model.sample_action(observation, eps)
        prev_o = observation
        observation, reward, done, _ = env.step(action)
        
        total_reward += reward
        if done and i<200:
            reward = -200
                    
        
        model.add_experience(prev_o, action, reward, observation, done)
        model.train(train_model)
        
        if i % copy_period == 0:
            #train_model.copy_from(model)
            train_model.load_from();
        i += 1


    return total_reward


def main():
    env = gym.make('CartPole-v0')
    gamma = 0.99
    copy_period = 50
    
    input_dim = len(env.observation_space.sample())
    output_dim = env.action_space.n
    h_sizes = [150,150]
    model = DeepQNetwork(input_dim, output_dim, h_sizes, gamma, "model")
    train_model = DeepQNetwork(input_dim, output_dim, h_sizes, gamma, "training_model")
    train_model.setup_copy_from(model)
    init = tf.global_variables_initializer()
    
    
    
    session = tf.InteractiveSession()
    session.run(init)
    model.set_session(session)
    train_model.set_session(session)
    
    
    N=500
    total_rewards = np.empty(N)
    for i in range(N):
        eps = 1.0/np.sqrt(i+1)
        total_reward = play_one(env, model, train_model, eps, copy_period)
        total_rewards[i] = total_reward
        if i % 25 ==0:
            last_100_r = total_rewards[max(0,i-100):i].mean()
            print("episode number: %s, last_100_reward_avg: %s, last play %s" 
                  %( i, last_100_r, total_reward))
    
    plot_x(total_rewards, 'rewards')
    plot_running_avg(total_rewards)
    
if __name__ == '__main__':
    main()
        
        
        
        