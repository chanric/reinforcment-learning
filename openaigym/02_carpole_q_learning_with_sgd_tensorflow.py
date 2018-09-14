#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 15:57:35 2018

@author: chari11
"""

import tensorflow as tf
from carpole_train_with_sgd import train_carpole_with_regressor


class TensorFlowSGDRegressor:
    def __init__(self, dimensions, lr=0.1):
        self._w = tf.Variable(tf.random_normal(shape=(dimensions,1), name='w'))
        self._X = tf.placeholder(tf.float32, shape=(None, dimensions), name='X')
        self._y = tf.placeholder(tf.float32, shape=(None,), name='y')
        
        #create calc fun
        y_hat = tf.reshape( tf.matmul(self._X, self._w), [-1])
        delta = y_hat - self._y
        cost = tf.reduce_sum(delta*delta)
        
        #create func
        self._train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)
        self._predict_op = y_hat
        
        #tensor flow init
        init = tf.global_variables_initializer()
        self._session = tf.InteractiveSession()
        self._session.run(init)
        
    def partial_fit(self, X, y):
        self._session.run(self._train_op, feed_dict={self._X: X, self._y: y})
    
    def predict(self, X):
        return self._session.run(self._predict_op, feed_dict={self._X: X})
    
class TFSGDFactory:
    @staticmethod
    def create(dimensions, lr=0.1):
        return TensorFlowSGDRegressor(dimensions, lr)


if __name__ == '__main__':
    train_carpole_with_regressor(TFSGDFactory)