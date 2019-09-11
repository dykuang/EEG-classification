# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 16:26:34 2019

@author: dykua

Some Utility functions
"""
import numpy as np

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
def generator(X, Y, batchsize, shift_limit = 5):
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
        
        if prob < 3:
            rand_shift = np.random.choice(shift_limit*2) - shift_limit  
            X_gen = np.roll(X_gen, rand_shift, axis = -2)                              
#            yield [np.roll(X_gen, rand_shift, axis = -2) , Y_gen] 
        
#        else:
#            yield [X_gen, Y_gen] 
        
        yield [ X_gen, Y_gen.toarray() ] # sparse OH matrix causing errors in keras fit. set input(sparse = True) does not seem to solve it.
        
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
    




