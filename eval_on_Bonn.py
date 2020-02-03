# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 11:29:10 2019

@author: dykua

For the evaluation on Bonn dataset

5-folds classification
"""

from keras.optimizers import Adam, SGD
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report


Params = {
        'batchsize': 32,
        'epochs': 100,
        'lr': 1e-4,
        }

'''
Load data
'''
#Xtrain = np.load(r'../MI_train.npy')
#Xtest = np.load(r'../MI_test.npy')
#Ytrain = np.load(r'../MI_train_D1_label.npy')
#Ytest = np.load(r'../MI_test_D1_label.npy')

#Xtrain = np.load(r'../fc_train.npy')
#Xtest = np.load(r'../fc_test.npy')
#
#Xtrain = np.load(r'../Xtrain.npy')
#Xtest = np.load(r'../Xtest.npy')
#Ytrain = np.load(r'../Ytrain1.npy')
#Ytest = np.load(r'../Ytest1.npy')

# ============== BCI-IV-set 3 ======================================
#dataset = 'D:/EEG/archive/BCI-IV-dataset3/'
#subject = 2
#Xtrain = np.load(dataset+r'S{}train.npy'.format(subject))
#Xtest = np.load(dataset+r'S{}test.npy'.format(subject))
#Ytrain = np.load(dataset+r'Ytrain.npy'.format(subject))
#Ytest = np.load(dataset+r'S{}Ytest.npy'.format(subject))[0]
# ===================================================================


#========= Bonn ==================================
# Can do a binary classification
#===================================================
dataset = 'D:/EEG/archive/Bonn/'


A = np.load(dataset + 'Z.npy')[...,None] ## healthy, eyes open
B = np.load(dataset + 'O.npy')[...,None] ## healthy, eyes closed
C = np.load(dataset + 'N.npy')[...,None] ## unhealthy, seizure free, not onsite
D = np.load(dataset + 'F.npy')[...,None] ## unhealthy, seizure free, onsite
E = np.load(dataset + 'S.npy')[...,None] ## unhealthy, seizure

X = np.vstack([B, D, E])
Y = np.hstack([np.zeros(len(B)),np.ones(len(D)), 2*np.ones(len(E))])

#from sklearn.preprocessing import MinMaxScaler, StandardScaler
#from preprocess import normalize_in_time, normalize_samples, rolling_max, normalize_mvar, Buterworth_batch
from scipy.stats import zscore

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10, random_state=123)
indexes = skf.split(X, Y)

from architectures import My_eeg_net_1d as eeg_net
CM = []
report = []
for train_index, test_index in indexes:
    Xtrain, Xtest = X[train_index], X[test_index]
    Ytrain, Ytest = Y[train_index], Y[test_index]
    
    '''
    Normalize data
    '''


    X_train_transformed = zscore(Xtrain, axis=1)
    X_test_transformed = zscore(Xtest, axis=1)
    #
    
    Params['samples'], Params['t-length'], Params['feature dim'] = X_train_transformed.shape
    
    '''
    One-Hot labels
    '''
    enc = OneHotEncoder()
    Ytrain_OH = enc.fit_transform(Ytrain[:,None])
    Ytest_OH = enc.transform(Ytest[:,None])
    _, Params['n classes'] = Ytrain_OH.shape


    '''
    Build Model
    '''
        
    
    Mymodel = eeg_net(Params['n classes'], Chans = Params['feature dim'], 
                          Samples = Params['t-length'], 
                          dropoutRate = 0.5, kernLength = 50, F1 = 32, 
                          D = 2, F2 = 64, norm_rate = 0.25, 
                          optimizer = Adam,
                          learning_rate=Params['lr'],
                          dropoutType = 'Dropout',
                          act = 'softmax')
    

    '''
    Train Model
    '''
    hist = Mymodel.fit(X_train_transformed, Ytrain_OH, 
                epochs=Params['epochs'], batch_size = Params['batchsize'],
                validation_data = (X_test_transformed, Ytest_OH),
#                validation_split=0.2,
                verbose=1,
                callbacks=[],
#                class_weight = weight_dict
                )

    '''
    Summary statistics
    '''
#    pred_train = Mymodel.predict(X_train_transformed)
#    print("Acc on trained data: {}".format(accuracy_score(Ytrain, np.argmax(pred_train, axis=1))))
    pred_test = Mymodel.predict(X_test_transformed)
#    print("Acc on test data: {}".format(accuracy_score(Ytest, np.argmax(pred_test, axis=1))))
    CM.append(confusion_matrix(Ytest, np.argmax(pred_test, axis=1)))
#    print(confusion_matrix(Ytest, np.argmax(pred_test, axis=1)) )
    report.append(classification_report(Ytest, np.argmax(pred_test, axis=1)))

