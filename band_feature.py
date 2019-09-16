# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 17:50:50 2019

@author: dykua


Test the feature of relative band energy
"""
import numpy as np
Params = {
        'cut_off freq': 0.1,
        }

'''
Load data
'''
#Xtrain = np.load(r'../MI_train.npy')[:,:1000,:]
#Xtest = np.load(r'../MI_test.npy')[:,:1000,:]
#Ytrain = np.load(r'../MI_train_D1_label.npy')
#Ytest = np.load(r'../MI_test_D1_label.npy')
#
Xtrain = np.load(r'../Xtrain.npy')
Xtest = np.load(r'../Xtest.npy')
Ytrain = np.load(r'../Ytrain1.npy')
Ytest = np.load(r'../Ytest1.npy')
'''
Normalize data
'''

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from preprocess import normalize_in_time, normalize_samples, rolling_max, normalize_mvar, Buterworth_batch

X_train_transformed = Buterworth_batch(Xtrain, cut_off_freq = Params['cut_off freq'])
X_test_transformed = Buterworth_batch(Xtest, cut_off_freq = Params['cut_off freq'])

from preprocess import band_decompose

#Params['samples'], Params['t-length'], Params['feature dim'] = X_train_transformed.shape
X_train_transformed = band_decompose(X_train_transformed)
X_test_transformed = band_decompose(X_test_transformed)



X_train_features = np.reshape(X_train_transformed, (X_train_transformed.shape[0], -1) )
X_test_features = np.reshape(X_test_transformed, (X_test_transformed.shape[0], -1) )

#X_train_transformed, X_test_transformed, _ = normalize_samples(Xtrain, Xtest, MinMaxScaler, 0, 1)

'''
SVM
'''
from sklearn import svm
from sklearn.metrics import accuracy_score
clf = svm.SVC(C=1.0, gamma=0.001, decision_function_shape='ovr', 
              kernel = 'rbf', degree=3,
              class_weight=None, tol=1e-3)

clf.fit(X_train_features, Ytrain)
print("Acc with SVM -- train: {}".format(accuracy_score(clf.predict(X_train_features), Ytrain)))
print("Acc with SVM -- test: {}".format(accuracy_score(clf.predict(X_test_features), Ytest)))

