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
plt.plot(t, b, alpha=0.8)
plt.ylim([-3,5])
plt.grid()
#plt.title('Healthy')
plt.subplot(3,1,2)
plt.plot(t, d, 'darkorange',alpha=0.8)
plt.ylim([-3,5])
plt.grid()
#plt.title('Preictal')
plt.subplot(3,1,3)
plt.plot(t, e, 'g', alpha=0.8)
plt.ylim([-3,5])
plt.grid()
#plt.title('Seizure')

from scipy.signal import periodogram
plt.figure()
f_S1, p_S1=periodogram(b, fs=173.61, window=None, nfft=None, 
                       detrend='constant', return_onesided=True, scaling='density', 
                       axis=0)
plt.plot(f_S1, p_S1, label='Helathy')
f_S1, p_S1=periodogram(d, fs=173.61, window=None, nfft=None, 
                       detrend='constant', return_onesided=True, scaling='density', 
                       axis=0)
plt.plot(f_S1, p_S1, label='Preictal')
f_S1, p_S1=periodogram(e, fs=173.61, window=None, nfft=None, 
                       detrend='constant', return_onesided=True, scaling='density', 
                       axis=0)
plt.plot(f_S1, p_S1, label='Seizure')
plt.xlabel('Frequency(Hz)')      
plt.xlim(0,40)
plt.ylabel('V**2/Hz')
plt.legend()



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


# performance with different bandpassed filter

'''
BCI-dataset 3
'''
high_cut = np.array([199.99, 99.99, 49.99, 24.99, 12.99, 6.99])
S1_train = np.array([99.38, 99.38, 98.75, 98.75, 100, 63.75]) 
S1_test = np.array([68.92, 67.57, 67.57, 68.92, 60.81, 16.22])
S2_train = np.array([94.38, 93.13,93.13,95.00, 93.13, 65.63]) 
S2_test = np.array([47.95,46.57,47.95,46.57,46.57,21.92])

plt.figure()
plt.plot(high_cut, S1_train, '-^',label='S1_train')
plt.plot(high_cut, S1_test, '-^', label='S1_test')
plt.plot(high_cut, S2_train, '-^', label='S2_train')
plt.plot(high_cut, S2_test, '-^', label='S2_test')

plt.xlim(200, 0)
plt.grid()
plt.legend(['S1_train', 'S1_test', 'S2_train', 'S2_test'])
plt.ylabel('Accuracy (%)')
plt.xlabel('Upper bound of frequency kept (Hz)')

'''
Bonn
'''

high_cut = np.array([0.5, 0.25, 0.125, 1/16.0])*173.61
H = np.array([42, 42, 31, 20])/44*100
P = np.array([37, 35, 37, 28])/39*100
S = np.array([33, 33, 22, 2])/37*100

plt.figure()
plt.plot(high_cut, H, '-^',label='Healthy')
plt.plot(high_cut, P, '-^',label='Preictal')
plt.plot(high_cut, S, '-^',label='Seizure')

plt.xlim(100, 0)
plt.grid()
plt.legend()
plt.ylabel('Accuracy (%)')
plt.xlabel('Upper bound of frequency kept (Hz)')



'''
fNIRs
'''
s = np.empty((30,5))
s99 = np.empty((30, 5))
s99_nobc = np.empty((30, 5))

from scipy.io import loadmat
dataset = 'C:/Users/dykua/matlab projects/BCI/'
sample = loadmat(dataset + 'x01.mat')['x'].transpose((2,0,1))
sample_m99 = loadmat(dataset + 'x01_m99.mat')['x'].transpose((2,0,1))
sample_m99_nobc = loadmat(dataset + 'x01_m99_nobc.mat')['x'].transpose((2,0,1))

plt.figure()
plt.plot(np.arange(347)/13.333, sample[0,:,0], label='Filtered, baseline corrected')
plt.plot(np.arange(347)/13.333, sample_m99[0,:,0], label='Unfiltered, baseline corrected')
plt.plot(np.arange(347)/13.333, sample_m99_nobc[0,:,0], label='Unfiltered and no baseline correction')
plt.legend()
plt.xlabel('Time (s)')

for i in range(1, 31):
    eva = np.load('Eval-S{:02d}.npy'.format(i))
    eva99 = np.load('Eval-S{:02d}-m99.npy'.format(i))
    eva99_nobc = np.load('Eval-S{:02d}-m99-nobc.npy'.format(i))
    s[i-1] = eva[:,0]
    s99[i-1] = eva99[:,0]
    s99_nobc[i-1] = eva99_nobc[:,0]
    
print('Summary accuracy: mean: {} \n std:{} \n'.format(np.mean(s), np.std(np.mean(s,1))))
print('Summary accuracy m99: mean: {} \n std:{} \n'.format(np.mean(s99), np.std(np.mean(s99,1)))) 
print('Summary accuracy m99_nobc: mean: {} \n std:{} \n'.format(np.mean(s99_nobc), np.std(np.mean(s99_nobc,1)))) 
# plt.boxplot(s.transpose(), notch=False, whis=1.5, bootstrap=2000)


'''
Paired t-test 
'''
from scipy.stat import ttest_rel, ttest_ind
ttest_rel(np.mean(s, 1), np.mean(s99, 1))
ttest_rel(np.mean(s, 1), np.mean(s99_nobc, 1))
ttest_rel(np.mean(s99, 1), np.mean(s99_nobc, 1))


plt.figure()
plt.bar(range(1,31), np.mean(s, 1)*100, yerr= np.std(s, 1)*100,align='center', alpha=0.9, ecolor='black', capsize=2)
plt.bar(range(31,32), np.mean(s)*100, yerr= np.std(np.mean(s, 1))*100,align='center', alpha=0.9, ecolor='black', capsize=2)
plt.plot(range(32), 100*np.mean(s)*np.ones(32), '--')
plt.plot(range(32), 100*(np.mean(s)*np.ones(32) + np.std(np.mean(s,1))), 'g--')
plt.plot(range(32), 100*(np.mean(s)*np.ones(32) - np.std(np.mean(s,1))), 'g--')
plt.plot(range(32), 70.4*np.ones(32), 'r--')
plt.xlabel('Volunteers')
plt.ylabel('Accuracy (%)')
plt.text(0, 100, 'Grand mean: {:.02f}$\pm${:.02f}'.format(100*np.mean(s), 100*np.std(np.mean(s,1))))
plt.title('With filtered and baseline corrected data.')

plt.figure()
plt.bar(range(1,31), np.mean(s99, 1)*100, yerr= np.std(s99, 1)*100,align='center', alpha=0.9, ecolor='black', capsize=2)
plt.bar(range(31,32), np.mean(s99)*100, yerr= np.std(np.mean(s99, 1))*100,align='center', alpha=0.9, ecolor='black', capsize=2)
plt.plot(range(32), 100*np.mean(s99)*np.ones(32), '--')
plt.plot(range(32), 100*(np.mean(s99)*np.ones(32) + np.std(np.mean(s99,1))), 'g--')
plt.plot(range(32), 100*(np.mean(s99)*np.ones(32) - np.std(np.mean(s99,1))), 'g--')
plt.plot(range(32), 70.4*np.ones(32), 'r--')
plt.xlabel('Volunteers')
plt.ylabel('Accuracy (%)')
plt.text(0, 100, 'Grand mean: {:.02f}$\pm${:.02f}'.format(100*np.mean(s99), 100*np.std(np.mean(s99,1))))
plt.title('With unfiltered but baseline corrected data.')

plt.figure()
plt.bar(range(1,31), np.mean(s99_nobc, 1)*100, yerr= np.std(s99_nobc, 1)*100,align='center', alpha=0.9, ecolor='black', capsize=2)
plt.bar(range(31,32), np.mean(s99_nobc)*100, yerr= np.std(np.mean(s99_nobc, 1))*100,align='center', alpha=0.9, ecolor='black', capsize=2)
plt.plot(range(32), 100*np.mean(s99_nobc)*np.ones(32), '--')
plt.plot(range(32), 100*(np.mean(s99_nobc)*np.ones(32) + np.std(np.mean(s99_nobc,1))), 'g--')
plt.plot(range(32), 100*(np.mean(s99_nobc)*np.ones(32) - np.std(np.mean(s99_nobc,1))), 'g--')
plt.plot(range(32), 70.4*np.ones(32), 'r--')
plt.text(0, 100, 'Grand mean: {:.02f}$\pm${:.02f}'.format(100*np.mean(s99_nobc), 100*np.std(np.mean(s99_nobc,1))))
plt.title('With unfiltered and uncorrected data.')
plt.xlabel('Volunteers')
plt.ylabel('Accuracy (%)')

# plt.savefig('fnirs-eval', dpi=600, format='eps')
    
    

    






