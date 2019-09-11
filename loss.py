#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 16:14:28 2019

@author: dykuang

A collection of some loss functions
"""
import tensorflow as tf
def cross_reg(YTrue, YPred):
    '''
    For regularize the displacement in resample
    YPred: the displacement field with shape (sample, time, channel)
    '''
    diff = YPred[:,1:,:]-YPred[:,:-1,:] + 1
    abs_diff = tf.abs(diff)
    
    return tf.reduce_sum( tf.reshape(0.5*(abs_diff - diff), (-1, YPred.shape[-1]) ) )
    
    
