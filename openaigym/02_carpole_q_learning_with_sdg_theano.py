#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 12:32:09 2018

@author: chari11
"""

import theano
import theano.tensor as T
import numpy as np
from carpole_train_with_sgd import train_carpole_with_regressor

class TheanoSGDRegressor:
    def __init__(self, dimensions, lr=0.1):
        w = np.random.randn(dimensions)/np.sqrt(dimensions)
        self._w = theano.shared(w)
        #self._b = theano.shared(0., name="b")
        self._lr = lr
        
        #print(self._b.get_value())
        
        X = T.matrix('X')
        y = T.vector('y')
        y_hat = X.dot(self._w)
        delta = y - y_hat
        cost = delta.dot(delta)
        grad = T.grad(cost, self._w)
        updates = [(self._w, self._w - self._lr*grad)]
        
        self._train_op = theano.function(
                inputs=[X, y],
                updates=updates)
        self._predict_op = theano.function(
                inputs=[X],
                outputs=y_hat)
        
    def partial_fit(self, X, y):
        self._train_op(X,y)
        
    def predict(self, X):
        return self._predict_op(X)

class TheanoSDGFactory:
    @staticmethod
    def create(dimensions):
        return TheanoSGDRegressor(dimensions)
    
if __name__ == '__main__':
    train_carpole_with_regressor(TheanoSDGFactory)