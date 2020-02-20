# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 16:26:34 2019

@author: dykua

Some Utility functions
"""
import numpy as np
import keras.backend as K

def get_grad_tensor(outs, ins, order):
    '''
    getting the gradient of outs wrt ins till order 
    '''
    grad_tensor = [K.gradients(outs, ins)]
    for o in range(1, order):
        grad_tensor.append(K.gradients(grad_tensor[-1], ins))
        
    return grad_tensor

def get_grad_val(outs, ins, order, eval_at):
    '''
    Actually compute the value of grad tensors evaluated at 'eval_at'
    '''
    grad_tensor = get_grad_tensor(outs, ins, order)
    grad_func = K.function([ins], [tensor[0] for tensor in grad_tensor])
    grad_val = grad_func([eval_at])
    
    return grad_val 

def save_model(Model, filename='./model.h5', weights_only = True):
    if weights_only:
        Model.save_weights(filename)
        print('Weights saved.')
    else:
        Model.save(filename)
        print('Model saved.')
        

def Get_batch(X,Y,batchsize):
    '''
    Get a random batch from training data X, Y
    '''
    n_samples = X.shape[0]
    ind = np.random.choice(len(n_samples), batchsize, replace=False)
    X_batch = X[ind]
    Y_batch = Y[ind]
        
    return [X_batch, Y_batch]

def summarize_performance(model, step):
    filename = 'weights_at_step_{}.h5'.format(step)
    model.save_weights(filename)
    print("Weights saved at step {}.".format(step))
    
## Train with fit_generator
def generator(X, Y, batchsize, shift_limit = 5, freq = 4):
    '''
    A generator for keras' training with "fit_generator"
    '''  
    
    X_gen = np.empty([batchsize] + list(X.shape[1:]))
    Y_gen = np.empty([batchsize] + list(Y.shape[1:]))
    n_samples = X.shape[0]
    while True:
        prob = np.random.choice(10, 1)
        ind = np.random.choice(n_samples, batchsize, replace=False)
        X_gen = X[ind]
        Y_gen = Y[ind]
        
        if prob < freq:
            rand_shift = np.random.choice(shift_limit*2) - shift_limit  
            X_gen = np.roll(X_gen, rand_shift, axis = -2)                              
#            yield [np.roll(X_gen, rand_shift, axis = -2) , Y_gen] 
        
#        else:
#            yield [X_gen, Y_gen] 
        
        yield [ X_gen, Y_gen.toarray() ] # sparse OH matrix causing errors in keras fit. set input(sparse = True) does not seem to solve it.


def generator_freq(X, Y, batchsize, bandwidth = 30, dist_to_end = 50, freq = 5):
    '''
    augment input with cropped frequency 
    '''  
    
    X_gen = np.empty([batchsize] + list(X.shape[1:]))
    Y_gen = np.empty([batchsize] + list(Y.shape[1:]))
    n_samples = X.shape[0]
    while True:
        prob = np.random.choice(10, 1)
        start = int(np.random.choice(X.shape[1]//2 - dist_to_end, 1))
        end = start + bandwidth
        ind = np.random.choice(n_samples, batchsize, replace=False)
        X_gen = X[ind]
        Y_gen = Y[ind]
        
        if prob < freq:
            X_fft = np.fft.rfft(X_gen, axis=1)
            X_fft[:, start:end, :] = 0              
            X_gen = np.fft.irfft(X_fft, axis=1)                             
        
        yield [ X_gen, Y_gen.toarray()]
        

from preprocess import batch_band_pass
def generator_filter(X, Y, batchsize, min_freq_cut, max_freq_cut, samplingfreq, strength=0.5):
    '''
    augment input with low pass filter
    '''  
    
    X_gen = np.empty([batchsize] + list(X.shape[1:]))
    Y_gen = np.empty([batchsize] + list(Y.shape[1:]))
    n_samples = X.shape[0]
    while True:
        prob = np.random.uniform(0,1)
        ind = np.random.choice(n_samples, batchsize, replace=False)
        X_gen = X[ind]
        Y_gen = Y[ind]
        
        if prob < strength:
              freq_cut = np.random.uniform(min_freq_cut, max_freq_cut)
              X_gen = batch_band_pass(X_gen, 0.01, freq_cut, samplingfreq)              
        
        yield [ X_gen, Y_gen.toarray()]

        
## Train with custom Train_on_batch method, did not test it, may be slow
def train(model, X, Y, num_steps, batchsize, check_point=100):
    log=[]
    n_samples = X.shape[0]
    for step in range(num_steps): # put a generator here to fasten the code?
        print("on step {} ===========>".format(step))
        ind = np.random.choice(len(n_samples), batchsize, replace=False)
        X_batch = X[ind]
        Y_batch = Y[ind]
        loss =  model.train_on_batch(X_batch, Y_batch)     
        log.append(loss) 
               
        if (step +1) % check_point == 0:
            print("step {}: Total loss: {};".format(step, loss))
            summarize_performance(model, step)
            
    return log    
    

## ===================================================
#  For evaluations
## ======================================================
def scores(CM):
    '''
    Calculate the accuracy, precision, recall, specificity and f1_score from confusion matrix
    It returns a matrix with class labels as rows and scores above as columns
    '''
    n_classes = CM.shape[0]
    scores = np.empty((n_classes, 5))
    _sum = np.sum(CM)
    col_sum = np.sum(CM, axis=0)
    row_sum = np.sum(CM, axis=1)
    TP = CM.diagonal()
    scores[:,0] = TP/_sum # accuracy per class, sum up them for accuracy
    scores[:,1] = TP/col_sum # precision per class
    scores[:,2] = TP/row_sum  # recall per class 
    scores[:,3] = (_sum-row_sum-col_sum+TP)/(_sum-row_sum)
    scores[:,-1] = 2*(scores[:,1]*scores[:,2])/(scores[:,1]+scores[:,2]) # F1_score
    
    weight = row_sum/_sum
    weighted_summary = np.array([np.sum(scores[:,0]),
                                 np.dot(weight, scores[:,1]), 
                                 np.dot(weight, scores[:,2]),
                                 np.dot(weight, scores[:,3]),
                                 np.dot(weight, scores[:,-1])])
#    print(weight)
    return scores, weighted_summary

