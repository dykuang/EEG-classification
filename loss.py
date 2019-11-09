#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 16:14:28 2019

@author: dykuang

A collection of some loss functions
"""
import tensorflow as tf
from keras.losses import mean_absolute_error, categorical_crossentropy
def cross_reg(YTrue, YPred):
    '''
    For regularize the displacement in resample
    YPred: the displacement field with shape (sample, time, channel)
    '''
    diff = YPred[:,1:,:]-YPred[:,:-1,:] + 1
    abs_diff = tf.abs(diff)
    
    return tf.reduce_sum( tf.reshape(0.5*(abs_diff - diff), (-1, YPred.shape[-1]) ) )

def myLoss(weights=[1.0,1.0,1.0]): 
    def loss(yTrue, yPred):
#        z_mu, z_log_sigma, rec = yPred # custom loss receives tensors, this will not work
#        assert isinstance(yPred, list)
        
        z_mu = yPred[0]
        z_log_sigma = yPred[1]
        rec = yPred[2]
        
        print(z_mu.shape)
        
        reconstruction_loss = mean_absolute_error(yTrue[-1], rec)
        
        kl_loss = 1 + z_log_sigma - tf.square(z_mu) - tf.exp(z_log_sigma)
        kl_loss = tf.reduce_sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        kl_loss = tf.reduce_mean(kl_loss)
        
        clf_loss = categorical_crossentropy(yTrue[0], z_mu)
        
        return weights[0]*reconstruction_loss + weights[1]*kl_loss + weights[2]*clf_loss
    
    return loss
    
def KL_loss(z_mu, z_log_sigma): 
    def loss(yTrue, yPred):
        
        kl_loss = 1 + z_log_sigma - tf.square(z_mu) - tf.exp(z_log_sigma)
        kl_loss = tf.reduce_sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        kl_loss = tf.reduce_mean(kl_loss)
               
        return kl_loss
    
    return loss    
