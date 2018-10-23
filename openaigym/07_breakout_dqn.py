#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

G-one_hot(actions, output_dim)

"""

import gym
import tensorflow as tf
from scipy.misc import imresize
import numpy as np
import random
from datetime import datetime


MAX_EXPERIENCES = 500000
MIN_EXPERIENCES = 50000
TARGET_UPDATE_PERIOD = 10000
IM_SIZE = 80

def downsample_image(A):
    B = A[31:195]
    B = B.mean(axis=2) #grey scale
    
    return imresize(B, size=(IM_SIZE, IM_SIZE), interp='nearest')

def obs_to_new_state(obs):
    obs_small = downsample_image(obs)
    return np.stack([obs_small] * 4, axis=0)
    
def update_state(s, observation):
    obs = downsample_image(observation)
    return np.append(s[1:], np.expand_dims(obs, 0), axis=0)

class DeepQNetwork:
    def __init__(self, output_dim, conv_layer_sizes, hidden_layer_sizes, gamma, scope):
        self._output_dim = output_dim
        self.scope = scope
        
        
        with tf.variable_scope(scope):
            
            self._X = tf.placeholder(tf.float32, shape=(None, 4, IM_SIZE, IM_SIZE), name='X')
            self._G = tf.placeholder(tf.float32, shape=(None,), name='G')
            self._actions = tf.placeholder(tf.int32, shape=(None,), name='actions')
            
            Z = self._X/255
            Z = tf.transpose(Z, [0, 2, 3, 1]) #some use RGB, or GBR or whatever
            for num_out_filters, filters, pools in conv_layer_sizes:
                Z = tf.contrib.layers.conv2d(
                        Z,
                        num_out_filters,
                        filters,
                        pools,
                        activation_fn=tf.nn.relu)
            
            Z = tf.contrib.layers.flatten(Z)
            for size in hidden_layer_sizes:
                Z = tf.contrib.layers.fully_connected(Z, size)
            
            self._predict_op = tf.contrib.layers.fully_connected(Z, output_dim)
            
            selected_action_values = tf.reduce_sum(
                    self._predict_op * tf.one_hot(self._actions, output_dim),
                    reduction_indices=[1])
            
            
            cost = tf.reduce_mean(tf.square(self._G - selected_action_values))
            
            self._train_op = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(cost)
            self._cost = cost
            
    def set_session(self, session):
        self._session = session
    
    
    def copy_from(self, other):
        mine = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        mine = sorted(mine, key=lambda v: v.name)
        theirs = [t for t in tf.trainable_variables() if t.name.startswith(other.scope)]
        theirs = sorted(theirs, key=lambda v: v.name)
    
        ops = []
        for p, q in zip(mine, theirs):
          actual = self._session.run(q)
          op = p.assign(actual)
          ops.append(op)
    
        self._session.run(ops)  
    
    
    def predict(self, X):
        return self._session.run(self._predict_op, feed_dict={self._X: X});
    
    def update(self, X, actions, targets):
        cost, _ = self._session.run([self._cost, self._train_op],
                                    feed_dict={self._X: X,
                                             self._G: targets,
                                             self._actions: actions})
        return cost
    
    
    def sample_action(self, X, eps):
        if np.random.random() > eps:
            return np.argmax(self.predict([X])[0])
        else:
            return np.random.choice(self._output_dim)
    
    
def learn(model, target_model, experience_replay_buffer, gamma, batch_size):
        samples = random.sample(experience_replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*samples))
        
        next_Qs = target_model.predict(next_states)
        next_Q = np.amax(next_Qs, axis=1)
        targets = rewards + np.invert(dones).astype(np.float32)*gamma * next_Q
        
        loss = model.update(states, actions, targets)
        return loss
    
def play_one(env, total_t, experience_replay_buffer,
             model, target_model, gamma,
             batch_size, epsilon, epsilon_change, epsilon_min):
    t0 = datetime.now()    
    obs = env.reset()
    #special start.  take state and use it 4x
    obs_small = downsample_image(obs)
    state = np.stack([obs_small]*4, axis=0)
    done = False
    total_time_training = 0
    num_steps_in_episode = 0
    total_reward = 0

    
    while not done:
        
        if total_t % TARGET_UPDATE_PERIOD == 0:
            cpT = datetime.now()  
            target_model.copy_from(model)
            print("copy time", datetime.now()-cpT)
            print("Copied model parameters to target network. toatl_t =%s, period = %s" %
                  (total_t, TARGET_UPDATE_PERIOD))
        
        action = model.sample_action(state, epsilon)
        obs, reward, done, _ = env.step(action)
        obs_small = downsample_image(obs)
        next_state = np.append(state[1:], np.expand_dims(obs_small, 0), axis=0)
        #assert(next_state.shape == (4, 80, 80))
        
        total_reward += reward
        
        if len(experience_replay_buffer) == MAX_EXPERIENCES:
            experience_replay_buffer.pop(0)
        
        experience_replay_buffer.append((state, action, reward, next_state, done))
        
        t0_2 = datetime.now()
        learn(model, target_model, experience_replay_buffer, gamma, batch_size)
        dt = datetime.now() - t0_2
        
        total_time_training += dt.total_seconds()
        num_steps_in_episode += 1
        
        state = next_state
        total_t += 1
        
        epsilon = max(epsilon - epsilon_change, epsilon_min)
    
    return total_t, total_reward, (datetime.now()-t0), num_steps_in_episode, total_time_training/num_steps_in_episode, epsilon
    

if __name__ == '__main__':
    conv_layer_sizes = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    hidden_layer_sizes = [512]
    gamma = 0.99
    batch_size = 32
    num_episodes = 10000
    total_t = 0
    action_space_size =4
    experience_replay_buffer = []
    episode_rewards = np.zeros(num_episodes)
    
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_change = (epsilon - epsilon_min) / MAX_EXPERIENCES
    
    env = gym.envs.make("Breakout-v0")
    
    model = DeepQNetwork(
        output_dim=action_space_size,
        conv_layer_sizes = conv_layer_sizes,
        hidden_layer_sizes=hidden_layer_sizes,
        gamma=gamma,
        scope="model")
    target_model = DeepQNetwork(
            output_dim = action_space_size,
            conv_layer_sizes=conv_layer_sizes,
            hidden_layer_sizes=hidden_layer_sizes,
            gamma = gamma,
            scope="target_model"
            )
    
    with tf.Session() as sess:
        model.set_session(sess)
        target_model.set_session(sess)
        sess.run(tf.global_variables_initializer())
        
        obs = env.reset()
        state = obs_to_new_state(obs)
        
        for i in range(MIN_EXPERIENCES):
            
            action = np.random.choice(action_space_size)
            obs, reward, done, _= env.step(action)
            next_state = update_state(state, obs)
            experience_replay_buffer.append((state,action, reward, next_state, done))
            
            if done:
                state = obs_to_new_state(obs)
            else:
                state = next_state
        
        for i in range(num_episodes):
            total_t, episode_reward, duration,num_steps_in_episode, time_per_step, epsilon = play_one(
                    env,
                    total_t,
                    experience_replay_buffer,
                    model,
                    target_model, 
                    gamma,
                    batch_size,
                    epsilon,
                    epsilon_change,
                    epsilon_min)
            episode_rewards[i] = episode_reward
            
            last_100_avg = episode_rewards[max(0, i - 100):1 + 1].mean()
            print("Episode: ", i, "Duration: ", duration,
                "Num steps:", num_steps_in_episode,
                "Reward:", episode_reward,
                "Training time per step:", "%.3f" % time_per_step,
                "Avg Reward (Last 100):", "%.3f" % last_100_avg,
                "Epsilon:", "%.3f" % epsilon)
        
        
        
            
    
    
    
