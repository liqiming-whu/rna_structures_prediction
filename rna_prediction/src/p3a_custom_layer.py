#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 20:54:29 2016

@author: gallusse
"""
#import theano
import theano.tensor as T
#import lasagne

from lasagne.layers.base import Layer

class GlobalPool1DLayer(Layer):
    """
    GlobalPool1DLayer(incoming,
    pool_function=theano.tensor.mean, **kwargs)
    Global 1-dimensional pooling layer
    This layer pools globally across one dimension, either the third or the forth.
    We assume the follwing dimensions: [minibatch, channels, x, y]
    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.
    pool_function : callable
        the pooling function to use. This defaults to `theano.tensor.mean`
        (i.e. mean-pooling) and can be replaced by any other aggregation
        function.
    axis : integer
        either 2 or 3
    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.
    """

    def __init__(self, incoming, pool_function=T.mean, axis=2, **kwargs):
        super(GlobalPool1DLayer, self).__init__(incoming, **kwargs)
        self.pool_function = pool_function
        self.axis = axis

    def get_output_shape_for(self, input_shape):
        #return input_shape[:2]
        # the dimension of the non-pooled axis describes the output-shape
        if self.axis == 2:
            return input_shape[:2] + (input_shape[3],)
        elif self.axis == 3:
            return input_shape[:2] + (input_shape[2],)
            
    def get_output_for(self, input, **kwargs):
        if self.axis == 2:
            #input_shape = input.shape()
            #print(input_shape)
            reshaped = T.swapaxes(input, axis1=2, axis2=3)
            return self.pool_function(reshaped, axis=2)
        elif self.axis == 3:
            return self.pool_function(input, axis=3)
        