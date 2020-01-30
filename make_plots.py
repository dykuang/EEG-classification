# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:53:20 2019

@author: dykua

Making plots for experiements
"""

import numpy as np
import matplotlib.pyplot as plt

from visual import score_bar, plot_confusion_matrix
from Utils import scores

from scipy.stats import zscore

'''
BCI-dataset3
'''

dataset = 'D:/EEG/archive/BCI-IV-dataset3/'
S1 = np.load(dataset+r'S1train.npy')
S2 = np.load(dataset+r'S2train.npy')

S1_sample = [zscore(S1[i],axis=1) for i in range(0,160,40)]
S2_sample = [zscore(S2[i],axis=1) for i in range(0,160,40)]

t = np.arange(0,1,1/400)
plt.figure()
plt.subplot(4,2,1)
plt.plot(t, S1_sample[0][:,0], 'b')
plt.xticks([])
plt.title('S1-Left')
plt.grid()
plt.subplot(4,2,2)
plt.plot(t, S2_sample[0][:,0], 'b')
plt.title('S2-Left')
plt.xticks([])
plt.grid()

plt.subplot(4,2,3)
plt.plot(t, S1_sample[1][:,0], 'g')
plt.title('S1-Right')
plt.xticks([])
plt.grid()
plt.subplot(4,2,4)
plt.plot(t,S2_sample[1][:,0], 'g')
plt.title('S2-Right')
plt.xticks([])
plt.grid()

plt.subplot(4,2,5)
plt.plot(t,S1_sample[2][:,0], 'k')
plt.title('S1-Away')
plt.xticks([])
plt.grid()
plt.subplot(4,2,6)
plt.plot(t,S2_sample[2][:,0], 'k')
plt.title('S2-Away')
plt.xticks([])
plt.grid()

plt.subplot(4,2,7)
plt.plot(t,S1_sample[3][:,0], 'r')
plt.title('S1-Towards')
plt.xticks([])
plt.grid()
plt.subplot(4,2,8)
plt.plot(t,S2_sample[3][:,0], 'r')
plt.title('S2-Towards')
plt.xticks([])
plt.grid()



s1_s = np.load('S1_Single.npy')
s2_s = np.load('S2_Single.npy')
#s1_D = np.load('S1_Dual.npy')
s1_D = np.load('s1_D_new.npy')
#s2_D = np.load('S2_Dual.npy')
s2_D = np.load('S2_D_new.npy')
data = [s1_s, s1_D, s2_s, s2_D]
# plot the confusion matrix
for cm in data:
    plot_confusion_matrix(np.sum(cm, axis=0),['L','R','A','T'])

summary = []

for cm in data:
    weighted = np.empty((5,5))
    temp = 0
    for i in range(5):
        per_run, weighted[i] = scores(cm[i])
        temp += per_run
    summary.append(temp/5)
    print(np.mean(weighted, axis=0))

score_bar([summary[0][:,-1], summary[1][:,-1]], 
          ['k', 'r'], 
          ['L','R','A','T'], 
          ['Single','Dual'], ylim=[0.6, 0.8], width=0.25, 
          figsize=(15,8))

score_bar([summary[2][:,-1], summary[3][:,-1]], 
          ['k', 'r'], 
          ['L','R','A','T'], 
          ['Single','Dual'], ylim=[0.2, 0.6], width=0.25, 
          figsize=(15,8))

'''
Bonn
'''

dataset = 'D:/EEG/archive/Bonn/'

A = np.load(dataset + 'Z.npy')[...,None] ## healthy, eyes open
B = np.load(dataset + 'O.npy')[...,None] ## healthy, eyes closed
C = np.load(dataset + 'N.npy')[...,None] ## unhealthy, seizure free, not onsite
D = np.load(dataset + 'F.npy')[...,None] ## unhealthy, seizure free, onsite
E = np.load(dataset + 'S.npy')[...,None] ## unhealthy, seizure


b = zscore(B[0])
d = zscore(D[0])
e = zscore(E[0])

t = np.arange(0, 23.6, 1/173.61)[:-1]
plt.figure()
plt.subplot(3,1,1)
plt.plot(t, b, 'b', alpha=0.8)
plt.ylim([-3,5])
plt.grid()
plt.title('Healthy')
plt.subplot(3,1,2)
plt.plot(t, d, 'k', alpha=0.8)
plt.ylim([-3,5])
plt.grid()
plt.title('Preictal')
plt.subplot(3,1,3)
plt.plot(t, e, 'r', alpha=0.8)
plt.ylim([-3,5])
plt.grid()
plt.title('Seizure')


base_cm = np.load('baseCM.npy')
comp_cm_folds = np.load('CM_10fold.npy')
comp_cm = np.sum(comp_cm_folds,axis=0)

base, _ = scores(base_cm)
comp, _ = scores(comp_cm)

# plot the confusion matrix
plot_confusion_matrix(base_cm,['Healthy','Preictal','Seizure'])
plot_confusion_matrix(comp_cm,['Healthy','Preictal','Seizure'])

# barplot for f1-score
score_bar([base[:,-1],comp[:,-1]], 
          ['k', 'r'], 
          ['Healthy', 'Preictal', 'Seizure'], 
          ['Baseline','Ours'], ylim=[0.8, 1.0], width=0.25, 
          figsize=(15,8))