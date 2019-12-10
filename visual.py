# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 15:18:23 2019

@author: dykua

This script contains some functions for visualization
"""

import numpy as np
import matplotlib.pyplot as plt

def check_pred(true, pred):   
    plt.figure()
    plt.plot(true, color='r', alpha = 0.3)
    plt.plot(pred)
    plt.title('prediction')


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
    
    tiled_Cam_Grad = np.repeat(Cam_Grad, 30, axis=0)
    tiled_Cam_Grad = tiled_Cam_Grad.reshape([Cam_Grad.shape[0], 30, 1]).transpose([1,0,2])
#    plt.figure()
    plt.imshow(tiled_Cam_Grad[...,0])
    seed_input = (seed_input-np.min(seed_input))/np.max(seed_input)
    plt.plot(seed_input*13)
#    plt.plot(seed_input)
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
         

def score_bar(datalist, colorlist, labellist, namelist,
              width=0.15, ylim = [0., 1.1], alpha = 0.5, figsize=(8,30),):
    '''
    datalist: a list of data to plot, each member is a numpy array
    colorlist: a list of color for each group member
    labellist: a list, name for each group
    namelist: a list, name for the legend
    '''
    # Setting the positions and width for the bars
    pos = list(range(len(labellist))) 
#    width = 0.1 
        
    # Plotting the bars
    fig, ax = plt.subplots(figsize=figsize)
    num_data = len(datalist)
    # Create a bar with pre_score data,
    # in position pos,
    for i in range(num_data):
        plt.bar([p+width*i for p in pos], datalist[i], 
                width=width, alpha=alpha, color=colorlist[i], label=namelist[i])
 
    
    # Set the y axis label
    ax.set_ylabel('score')
    ax.set_xlabel('classes')
    # Set the chart's title
    ax.set_title('Summaries')
    
    # Set the position of the x ticks
    ax.set_xticks([p + 1.0 * width for p in pos])
    
    # Set the labels for the x ticks
    ax.set_xticklabels(labellist)
    
    # Setting the x-axis and y-axis limits
    plt.xlim(min(pos)-width, max(pos)+2*width)
    plt.ylim(ylim)
    plt.xticks(rotation = 0)
    # Adding the legend and showing the plot
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()
       

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    cm: the confusion matrix
    classes: a list, name for each class
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix'

    # Compute confusion matrix
#    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
#    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
#    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True',
           xlabel='Predicted')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


