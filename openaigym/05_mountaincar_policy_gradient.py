#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

2 networks. 1 for Policy, 1 for Value.
Value Policy:
    Basic. Only needs feature transformer because we have a big space
Policy Model.  
    Fit uses advantage. And it uses a mean/standard dev
    Mean/dev shares the standard series of hidden layers but end in
        mean = layer with 1-1 act_func
        std_dev = layer with softmax (or use non negiative capping func)
    We then predict with
           mean /w std_v
           (clip for this openai env)
    Train with 

            log_probs = norm.log_prob(self._actions)
        cost = -tf.reduce_sum(self._advantages * log_probs + 0.1*norm.entropy())
        self._train_op = tf.train.AdamOptimizer(1e-3).minimize(cost)
            


"""
import numpy as np
import gym
import tensorflow as tf
from common_plots import plot_running_avg, plot_x
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler


class FeatureTransformer:
    def __init__(self, env, n_components=1000):
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        self._sc = StandardScaler()
        self._sc.fit(observation_examples)
        
        self._featurizer = FeatureUnion([
                ('rbf1', RBFSampler(gamma=5.0, n_components=n_components)),
                ('rbf2', RBFSampler(gamma=2.0, n_components=n_components)),
                ('rbf3', RBFSampler(gamma=1.0, n_components=n_components)),
                ('rbf4', RBFSampler(gamma=.5, n_components=n_components)),
                ])
        #do test
        samples = self._featurizer.fit_transform(self._sc.transform(observation_examples))
        self.dimensions = samples.shape[1]
    
    def transform(self, observations):
        scaled_o = self._sc.transform(observations)
        return self._featurizer.transform(scaled_o)
        

class HiddenLayer:
    def __init__(self, input_dim, output_dim, activation_func=tf.nn.tanh, zero_start=False):
        if zero_start:
            w = np.zeros((input_dim, output_dim), dtype=np.float32)
        else:
            w = tf.random_normal(shape=(input_dim, output_dim))*np.sqrt(2. / input_dim, dtype=np.float32)
        self._w = tf.Variable(w)
        self._b = tf.Variable(np.zeros(output_dim).astype(np.float32))
        self._act_f = activation_func
        
    def forward(self, X):
        a = tf.matmul(X,self._w) +self._b
        return self._act_f(a)

class PolicyModel:
    def __init__(self, dimensions, feature_transformer, hidden_layer_sizes=[]):
        self._ft = feature_transformer
        
        curr_in = dimensions
        self._layers = []
        for ls in hidden_layer_sizes:
            self._layers.append(HiddenLayer(curr_in, ls))
            curr_in = ls
        
        #final mean layer
        self._mean_layer = HiddenLayer(curr_in, 1, lambda x:x, zero_start=True)
        #final standard dev
        self._std_dev_layer= HiddenLayer(curr_in, 1, tf.nn.softplus)
        
        self._X = tf.placeholder(tf.float32, shape=(None, dimensions), name='X')
        self._actions = tf.placeholder(tf.float32, shape=(None,),name='actions')
        self._advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')
        
        Z = self._X
        for l in self._layers:
            Z = l.forward(Z)
        
        mean = self._mean_layer.forward(Z)
        #use 1e-5 to smooth out
        std_v = self._std_dev_layer.forward(Z) + 1e-5
        
        #reshape
        mean = tf.reshape(mean, [-1])
        std_v = tf.reshape(std_v, [-1])
        
        norm = tf.contrib.distributions.Normal(mean, std_v)
        #the clip is to make sure we stay within the valid input range
        self._predict_op = tf.clip_by_value(norm.sample(), -1, 1)
        
        log_probs = norm.log_prob(self._actions)
        cost = -tf.reduce_sum(self._advantages * log_probs + 0.1*norm.entropy())
        self._train_op = tf.train.AdamOptimizer(1e-3).minimize(cost)
        
    def set_session(self, session):
        self._session = session
    
    def partial_fit(self, X, actions, advantages):
        X = np.atleast_2d(X)
        X = self._ft.transform(X)
        actions = np.atleast_1d(actions)
        advantages = np.atleast_1d(advantages)
        
        self._session.run(self._train_op,
                          feed_dict={self._X: X,
                                     self._advantages: advantages,
                                     self._actions: actions})
    def predict(self, X):
        X = np.atleast_2d(X)
        X = self._ft.transform(X)
        return self._session.run(self._predict_op,
                                 feed_dict={self._X: X})[0]
        
    def sample_action(self,X):
        return self.predict(X)
    

class ValueModel:
    def __init__(self, dimensions, feature_transformer, hidden_layer_sizes=[]):
        self._ft = feature_transformer
        
        input_size = dimensions
        self._layers =  []
        for ls in hidden_layer_sizes:
            self._layers.append(HiddenLayer(input_size, ls))
            input_size = ls
        
        #final layer
        self._layers.append(HiddenLayer(input_size, 1, activation_func= lambda x:x))
        
        self._X = tf.placeholder(tf.float32, shape=(None, dimensions), name='X')
        self._y = tf.placeholder(tf.float32, shape=(None,), name='y')
        
        Z = self._X
        for l in self._layers:
            Z = l.forward(Z)
        y_hat = tf.reshape(Z, [-1])
        
        self._predict_op = y_hat
        
        cost = tf.reduce_sum(tf.square(y_hat - self._y))
        self._train_op = tf.train.AdamOptimizer(1e-1).minimize(cost)
    
    def set_session(self, session):
        self._session = session
    
    def partial_fit(self, X, y):
        X = np.atleast_2d(X)
        X = self._ft.transform(X)
        y = np.atleast_1d(y)
        self._session.run(self._train_op, feed_dict={self._X: X,
                                                     self._y: y})
    
    def predict(self, X):
        X = np.atleast_2d(X)
        X = self._ft.transform(X)
        return self._session.run(self._predict_op,
                          feed_dict={self._X: X})

def play_one_td(env, policy_model, value_model, gamma=0.99):
    observation = env.reset()
    total_rewards = 0
    i =0
    done = False
    
    while not done:
        action = policy_model.sample_action(observation)
        prev_o = observation
        observation, reward, done, _ = env.step([action])
        
        #update value model
        V_next = value_model.predict(observation)
        G = reward + gamma*V_next
        advantage = G - value_model.predict(prev_o)
        policy_model.partial_fit(prev_o, action, advantage)
        value_model.partial_fit(prev_o, G)
        
        total_rewards += reward
        i += 1
    
    return total_rewards, i

def main():
    gamma=0.98
    env = gym.make('MountainCarContinuous-v0')
    ft = FeatureTransformer(env, n_components=100)
    dim = ft.dimensions

    
    policy_model = PolicyModel(dim, ft, [])
    value_model = ValueModel(dim, ft, [])
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)
    policy_model.set_session(session)
    value_model.set_session(session)
    
    
    N = 50    
    total_rewards = np.empty(N)
    for i in range(N):
        total_reward, steps_num = play_one_td(env, policy_model, value_model, gamma)
        total_rewards[i] = total_reward
        if i %2 ==0:
            last_100_r = total_rewards[max(0,i-100):i].mean()
            print("episode number: %s, last_100_reward_avg: %s" %( i, last_100_r))
            
    plot_x(total_rewards, 'rewards')
    plot_running_avg(total_rewards)

if __name__ == '__main__':
    main()