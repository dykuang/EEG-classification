# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 15:18:23 2019

@author: dykua

This script contains some functions for visualization
"""

import numpy as np
import matplotlib.pyplot as plt
def vis_dist(X):
    '''
    Visualize the distribution of time series data X：
    X: (trials, recoded values)
    '''
    
    X_std = np.std(X, axis = 0)
    X_mean = np.mean(X, axis = 0)
    plt.figure()
    plt.plot(X_mean)
    plt.plot(X_mean - X_std, 'r')
    plt.plot(X_mean + X_std, 'r')
    plt.title("mean and 1 std shown.")
    
def rgb2gray(rgb, weight= 0):
    if weight:
        return np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721])
    else:
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

from vis.visualization import visualize_cam    
def CAM_on_input(model, layer_idx, filter_indices,
                  seed_input, penultimate_layer_idx=None,
                  backprop_modifier=None, grad_modifier=None,
                  gray = 0):
    '''
    For 1d singal, place the CAM on input
    '''
    
    Cam_Grad = visualize_cam(model, layer_idx, filter_indices, 
                             seed_input, penultimate_layer_idx,
                             backprop_modifier, grad_modifier)
    
    tiled_Cam_Grad = np.repeat(Cam_Grad, 25, axis=0)
    tiled_Cam_Grad = tiled_Cam_Grad.reshape([Cam_Grad.shape[0], 25, 3]).transpose([1,0,2])
#    plt.figure()
    plt.imshow(tiled_Cam_Grad)
    plt.plot(19-seed_input*19)
    if gray:
        plt.figure()
        plt.plot(rgb2gray(Cam_Grad))
        
def CAM_distr(model, layer_idx, filter_indices,
                  group_seed_input, penultimate_layer_idx=None,
                  backprop_modifier=None, grad_modifier=None,
                  gray = 0):
    '''
    Gather CAM for each sample with the sample label
    '''
    distr = []
    for seed_input in group_seed_input:
         distr.append(visualize_cam(model, layer_idx, filter_indices, 
                                 seed_input, penultimate_layer_idx,
                                 backprop_modifier, grad_modifier)
                      )
    distr = np.stack(distr)
    if gray:
        plt.imshow(distr, cmap='gray')
        
    plt.imshow(distr)
     
    return distr
         
       

