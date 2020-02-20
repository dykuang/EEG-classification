# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 15:45:05 2019

@author: dykua

Training script

Q:  30% gap accuracy between train and test
      More data --- data augmentation?
      A simpler Model? ---
      
      
ToDo: 
    
    * Try the rolling mean, more regularization.
    * Blend the training set of S1 and S2, possibly learning the "invariance"?
    * Any unsupervised ideas?

"""

from keras.optimizers import Adam, SGD
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

Params = {
        'batchsize': 32,
        'epochs': 30,
        'lr': 1e-4,
        'cut_off freq': 0.1,
        'num downsampling': None,
        'low cut': 0.01,
        'exp num': 3
        }

if Params['exp num'] == 1:
    Params['high cut'] = 199.99
    Params['sampling freq'] = 400
    dataset = 'D:/EEG/archive/BCI-IV-dataset3/'
    subject = 2
    Xtrain = np.load(dataset+r'S{}train.npy'.format(subject))
    Xtest = np.load(dataset+r'S{}test.npy'.format(subject))
    Ytrain = np.load(dataset+r'Ytrain.npy'.format(subject)) # ordering are the same for both subjects: 40 + 40 + 40+ 40
    Ytest = np.load(dataset+r'S{}Ytest.npy'.format(subject))[0]
    
    # Xtrain = np.load('D:/EEG/archive/extra_data/re_train.npy')
    # Xtest = np.load('D:/EEG/archive/extra_data/re_test.npy')
elif Params['exp num'] == 2:
    Params['high cut'] = 173.6/2,
    Params['sampling freq'] = 173.61
    dataset = 'D:/EEG/archive/Bonn/'
    
    A = np.load(dataset + 'Z.npy')[...,None] ## healthy, eyes open
    B = np.load(dataset + 'O.npy')[...,None] ## healthy, eyes closed
    C = np.load(dataset + 'N.npy')[...,None] ## unhealthy, seizure free, not onsite
    D = np.load(dataset + 'F.npy')[...,None] ## unhealthy, seizure free, onsite
    E = np.load(dataset + 'S.npy')[...,None] ## unhealthy, seizure
    
    X = np.vstack([B, D, E])
    Y = np.hstack([np.zeros(len(B)),np.ones(len(D)), 2*np.ones(len(E))])
    
    from sklearn.model_selection import train_test_split
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.4, random_state=42)    
elif Params['exp num'] == 3:
        Params['high cut'] = 13.3333//2
        Params['sampling freq'] = 13.3333
        from scipy.io import loadmat
        dataset = 'C:/Users/dykua/matlab projects/BCI/'
      
        
        # subject = 2
        # X = loadmat(dataset + 'x{:02d}.mat'.format(subject))['x'].transpose((2,0,1))
        # Y = loadmat(dataset + 'label{:02d}.mat'.format(subject))['y'].flatten()-1
        
        
        X = []
        Y = []
        for subject in range(1,31):
            X.append(loadmat(dataset + 'x{:02d}.mat'.format(subject))['x'].transpose((2,0,1)))
            Y.append(loadmat(dataset + 'label{:02d}.mat'.format(subject))['y'].flatten()-1 )
        X = np.vstack(X)
        Y = np.hstack(Y)
        
        from sklearn.utils import shuffle
        X, Y = shuffle(X, Y)
        
        from sklearn.model_selection import train_test_split
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(
                                        X, Y, test_size=0.75, random_state=42,
                                        shuffle = True, stratify = Y
                                        )

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
# =============================================================================
# dataset = 'D:/EEG/archive/BCI-IV-dataset3/'
# subject = 2
# Xtrain = np.load(dataset+r'S{}train.npy'.format(subject))
# Xtest = np.load(dataset+r'S{}test.npy'.format(subject))
# Ytrain = np.load(dataset+r'Ytrain.npy'.format(subject)) # ordering are the same for both subjects: 40 + 40 + 40+ 40
# Ytest = np.load(dataset+r'S{}Ytest.npy'.format(subject))[0]
# =============================================================================

# Xtrain = np.load('D:/EEG/archive/extra_data/re_train.npy')
# Xtest = np.load('D:/EEG/archive/extra_data/re_test.npy')

# ===================================================================




# ====================== RT/LT/FT fNIRs data ==================================
# =============================================================================
# from scipy.io import loadmat
# dataset = 'C:/Users/dykua/matlab projects/BCI/'
# 
# 
# # subject = 2
# # X = loadmat(dataset + 'x{:02d}.mat'.format(subject))['x'].transpose((2,0,1))
# # Y = loadmat(dataset + 'label{:02d}.mat'.format(subject))['y'].flatten()-1
# 
# 
# X = []
# Y = []
# for subject in range(1,31):
#     X.append(loadmat(dataset + 'x{:02d}.mat'.format(subject))['x'].transpose((2,0,1)))
#     Y.append(loadmat(dataset + 'label{:02d}.mat'.format(subject))['y'].flatten()-1 )
# X = np.vstack(X)
# Y = np.hstack(Y)
# 
# from sklearn.utils import shuffle
# X, Y = shuffle(X, Y)
# 
# from sklearn.model_selection import train_test_split
# Xtrain, Xtest, Ytrain, Ytest = train_test_split(
#                                 X, Y, test_size=0.75, random_state=42,
#                                 shuffle = True, stratify = Y
#                                 )
# 
# =============================================================================

# =============================================================================
# target_dir = 'C:/Users/dykua/Google Drive/Researches/Canada/The leaf project/data/leaf/'
# 
# leaf_data = np.load(target_dir + 'S_leaf_CCD.npy')
# leaf_label = np.load(target_dir + 'S_leaf_label.npy')
# size = 75*15
# 
# from sklearn.model_selection import train_test_split
# Xtrain, Xtest, Ytrain, Ytest , ind_train, ind_test = train_test_split(
#                              leaf_data, leaf_label, np.arange(size), 
#                              test_size=.1, 
#                              random_state = 233,
#                              shuffle = True, stratify = leaf_label)
# =============================================================================

#flag = 6
#target_dir = 'D:/EEG/archive/extra_data'
#keywords = ['IWS', 'Worms', 'Phoneme', 'ChlorineConcentratio', 'ElectricDevices', 'NN', 'NN_last', 
#            'NN_last_Pxx', 'NN_last_hist', 'NN_last_allC']
#Xtrain = np.load(target_dir+ '/' + keywords[flag]+'_X_train.npy')
#Xtest = np.load(target_dir + '/' + keywords[flag]+'_X_test.npy')
#Ytrain = np.load(target_dir + '/' + keywords[flag]+'_Y_train.npy')
#Ytest = np.load(target_dir + '/' + keywords[flag]+'_Y_test.npy')
#
#
#Xtrain= Xtrain[...,None]
#Xtest = Xtest[...,None]


#Xtrain = np.reshape(Xtrain, [160, -1, 10])
#Xtest = np.reshape(Xtest, [74, -1, 10])

#from preprocess import get_epdf, power_spectrum_batch
##
##Xtrain = get_epdf(Xtrain, (np.amin(Xtrain), np.amax(Xtrain)), 400)
##Xtest = get_epdf(Xtest, (np.amin(Xtrain), np.amax(Xtrain)), 400)
#
#Xtrain = power_spectrum_batch(Xtrain, Xtrain.shape[1], use_welch=True)
#Xtest = power_spectrum_batch(Xtest, Xtest.shape[1], use_welch=True)

#========= Bonn ==================================
# Can do a binary classification
#===================================================
# =============================================================================
# dataset = 'D:/EEG/archive/Bonn/'
# #H = np.load(dataset + 'Z.npy')[...,None]
# #Y_H = np.zeros(len(H))
# #U = np.load(dataset + 'N.npy')[...,None]
# #Y_U = np.ones(len(U))
# ##Y_U = np.zeros(len(U))
# #S = np.load(dataset + 'Se.npy')
# #Y_S = 2*np.ones(len(S))
# ##Y_S = np.ones(len(S))
# 
# #X = np.vstack([H,U,S])
# #Y = np.hstack([Y_H,Y_U,Y_S])
# 
# A = np.load(dataset + 'Z.npy')[...,None] ## healthy, eyes open
# B = np.load(dataset + 'O.npy')[...,None] ## healthy, eyes closed
# C = np.load(dataset + 'N.npy')[...,None] ## unhealthy, seizure free, not onsite
# D = np.load(dataset + 'F.npy')[...,None] ## unhealthy, seizure free, onsite
# E = np.load(dataset + 'S.npy')[...,None] ## unhealthy, seizure
# 
# #A = np.stack([A[:,i*400:(i+1)*400,:] for i in range(10)], axis=-1)[...,0,:]
# #B = np.stack([B[:,i*400:(i+1)*400,:] for i in range(10)], axis=-1)[...,0,:]
# #C = np.stack([C[:,i*400:(i+1)*400,:] for i in range(10)], axis=-1)[...,0,:]
# #D = np.stack([D[:,i*400:(i+1)*400,:] for i in range(10)], axis=-1)[...,0,:]
# #E = np.stack([E[:,i*400:(i+1)*400,:] for i in range(10)], axis=-1)[...,0,:]
# 
# #X = np.vstack([A,B,C,D,E])
# #Y = np.hstack([np.zeros(len(A)),np.ones(len(B)),2*np.ones(len(C)),3*np.ones(len(D)),4*np.ones(len(E))])
# 
# X = np.vstack([B, D, E])
# Y = np.hstack([np.zeros(len(B)),np.ones(len(D)), 2*np.ones(len(E))])
# 
# from sklearn.model_selection import train_test_split
# Xtrain, Xtest, Ytrain, Ytest = train_test_split(
#                                 X, Y, test_size=0.4, random_state=42)
# =============================================================================



'''
Normalize data
'''
#X_train_transformed = Xtrain[:,:75,:]
#X_test_transformed = Xtest[:,:75,:]
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from preprocess import normalize_in_time, normalize_samples, rolling_max, normalize_mvar, Buterworth_batch, band_pass, batch_band_pass

#X_train_transformed = Buterworth_batch(Xtrain, cut_off_freq = Params['cut_off freq']) # use this for non-smooth data
#X_test_transformed = Buterworth_batch(Xtest, cut_off_freq = Params['cut_off freq'])

# Xtrain =  batch_band_pass(Xtrain, Params['low cut'], Params['high cut'], Params['sampling freq'])
# Xtest =  batch_band_pass(Xtest, Params['low cut'], Params['high cut'], Params['sampling freq'])
       
from scipy.stats import zscore

#X_train_transformed = zscore(X_train_transformed, axis=1)
#X_test_transformed = zscore(X_test_transformed, axis=1)

X_train_transformed = zscore(Xtrain, axis=1)
X_test_transformed = zscore(Xtest, axis=1)

#X_train_transformed =  batch_band_pass(X_train_transformed, Params['low cut'], Params['high cut'], 400)
#X_train_transformed =  batch_band_pass(X_train_transformed, Params['low cut'], Params['high cut'], 400)


#X_train_transformed, X_test_transformed, _ = normalize_samples(X_train_transformed, X_test_transformed, MinMaxScaler, 0, 1)

#X_train_transformed, X_test_transformed = normalize_mvar(X_train_transformed, X_test_transformed)

#X_train_transformed = normalize_in_time(Xtrain,smooth = True, cut_off_freq = Params['cut_off freq'], order = 3)
#X_test_transformed = normalize_in_time(Xtest,smooth = True, cut_off_freq = Params['cut_off freq'], order = 3)
##X_train_transformed, X_test_transformed, _ = normalize_samples(X_train_transformed, X_test_transformed, MinMaxScaler, 0, 1)

#X_train_transformed = X_train_transformed[:,:,:2]
#X_test_transformed = X_test_transformed[:,:,:2]

#X_train_transformed = rolling_max(X_train_transformed, 30)
#X_test_transformed = rolling_max(X_test_transformed, 30)wwwwwwww

Params['samples'], Params['t-length'], Params['feature dim'] = X_train_transformed.shape

'''
One-Hot labels
'''
enc = OneHotEncoder()
Ytrain_OH = enc.fit_transform(Ytrain[:,None])
Ytest_OH = enc.transform(Ytest[:,None])
_, Params['n classes'] = Ytrain_OH.shape

'''
If NN's output is between -1 and 1 (tanh activation)
'''
#Ytrain_OH = 2*Ytrain_OH - np.ones(Ytrain_OH.shape)
#Ytest_OH = 2*Ytest_OH - np.ones(Ytest_OH.shape)
'''
Build Model
'''
_2D = False
if _2D:
    X_train_transformed = X_train_transformed[...,None]
    X_test_transformed = X_test_transformed[...,None]
    from architectures import My_eeg_net as eeg_net
#from architectures import TC_arc_sep as arc
#Mymodel = arc(Params['t-length'],
#                   Params['feature dim'],
#                   Params['n classes'],
#                   num_down = Params['num downsampling'],
#                   optimizer = Adam,
#                   learning_rate=Params['lr']
#                   )
else:
    from architectures import My_eeg_net_1d as eeg_net
    

Mymodel = eeg_net(Params['n classes'], Chans = Params['feature dim'], 
                      Samples = Params['t-length'], 
                      dropoutRate = 0.5, kernLength = 50, F1 = 32, 
                      D = 2, F2 = 64, norm_rate = 0.25, 
                      optimizer = Adam,
                      learning_rate=Params['lr'],
                      dropoutType = 'Dropout',
                      act = 'softmax')

#Mymodel = My_leafnet(Params['n classes'], length = Params['t-length'], poolsize=2)


Mymodel.summary()


from keras.callbacks import TensorBoard, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping
from datetime import datetime

'''
Some call back functions
'''
logdir = "tf-logs/"+datetime.now().strftime("%Y%m%d-%H%M%S")+'/'
tf_callback = TensorBoard(log_dir = logdir, 
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True,
                          write_grads=False,
                          batch_size = 32,
                          embeddings_freq=5, 
                          embeddings_layer_names=['dense'], 
                          embeddings_metadata=None, 
                          embeddings_data=X_test_transformed,
                          )

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5,
                              patience=5, min_lr=1e-6)

def myschedule(epochs, lr):
    new_lr = lr
    if epochs>50:
        new_lr = lr/2
    
    return new_lr

lr_schedule = LearningRateScheduler(myschedule)

ES = EarlyStopping(monitor='val_loss', 
                   min_delta=0.01, 
                   patience=6, 
                   verbose=0, mode='auto', baseline= None, restore_best_weights=True)

'''
Train Model
'''
randshift = False
randfreq = False
randfilter = True
aug = False
if aug:
    if randshift:
        from Utils import generator
        gen = generator(X_train_transformed, Ytrain_OH, Params['batchsize'], shift_limit = 40)
 
    if randfreq:
        from Utils import generator_freq
        gen = generator_freq(X_train_transformed, Ytrain_OH, Params['batchsize'], bandwidth = 50, dist_to_end = 100)

    if randfilter:
        from Utils import generator_filter
        gen = generator_filter(X_train_transformed, Ytrain_OH, Params['batchsize'],
                                           40, 60, Params['sampling freq'], strength=0.2)
        
        hist = Mymodel.fit_generator(gen, steps_per_epoch=Params['samples']//Params['batchsize'] + 1, 
                                     epochs=Params['epochs'], 
                                     verbose=1, callbacks=None, 
                                     validation_data=(X_test_transformed, Ytest_OH), 
                                     validation_steps=None, 
                                     class_weight=None, max_queue_size=10, 
                                     workers=1, use_multiprocessing=False, shuffle=True)
        
else:
#    weight_dict = np.array([1.0,1.0,1.0,1.0])
#    weight_dict = np.array([1.0,1.0,1.0, 1.0,1.0])
#    weight_dict = np.array([1.0, 1.0]) # for binary
    hist = Mymodel.fit(X_train_transformed, Ytrain_OH, 
                epochs=Params['epochs'], batch_size = Params['batchsize'],
                validation_data = (X_test_transformed, Ytest_OH),
#                validation_split=0.2,
                verbose=1,
                callbacks=[],
#                class_weight = weight_dict
                )

#Mymodel.save_weights('Mymodel_weights.h5')

'''
Visualize training history
'''
import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.figure()
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])

'''
Summary statistics
'''
print(Params)
from sklearn.metrics import accuracy_score
pred_train = Mymodel.predict(X_train_transformed)
print("Acc on trained data: {}".format(accuracy_score(Ytrain, np.argmax(pred_train, axis=1))))
pred_test = Mymodel.predict(X_test_transformed)
print("Acc on test data: {}".format(accuracy_score(Ytest, np.argmax(pred_test, axis=1))))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(Ytest, np.argmax(pred_test, axis=1)) )

#'''
#some visualizations of activations in conv layers
#'''
# import keras.backend as K
#act = K.function([Mymodel.layers[0].input], 
#                 [Mymodel.layers[6].output, 
#                  Mymodel.layers[8].output,
#                  Mymodel.layers[10].output,
#                  Mymodel.layers[12].output,
#                  Mymodel.layers[-3].output])
#_act = act([X_train_transformed[:1]])
#
## "image" activated function in each conv block
#plt.figure()
#for i in range(len(_act)-1):
#    plt.subplot(1,len(_act),i+1)
#    plt.imshow(_act[i][0], cmap='jet')
#    plt.xlabel('c')
#    plt.ylabel('time')
#
## first channel of activated function in each conv block
#plt.figure()
#for i in range(len(_act)-1):
#    plt.subplot(len(_act), 1, i+1)
#    plt.plot(_act[i][0,:,0])
#    plt.xlabel('time')
#
## first channel of each of the conv layer with different sizes
#plt.figure()
#for i in range(4):
#    plt.subplot(4, 1, i+1)
#    plt.plot(_act[0][0,:,i*16])
#    plt.xlabel('time')  
#    
    
# =============================================================================
# '''
# Forming a combined classifier: NN + SVM
# '''
# from sklearn import svm
# clf = svm.SVC(C=1.0, gamma=0.001, decision_function_shape='ovr', 
#               kernel = 'rbf', degree=3,
#               class_weight=None, tol=1e-3)
# 
# act = K.function([Mymodel.layers[0].input], [Mymodel.get_layer('dense').output])
# 
# feature = act([X_train_transformed])
# feature_test = act([X_test_transformed])
# to_svm = feature[-1]
# to_svm_test = feature_test[-1]
# 
# clf.fit(to_svm, Ytrain)
# pred_test_svm = clf.predict(to_svm_test)
# print("Acc with an extra SVM: {}".format(accuracy_score(pred_test_svm, Ytest)))
# print(confusion_matrix(Ytest, pred_test_svm) )
# =============================================================================

#from visual import CAM_on_input
#plt.figure()
#for i in range(4):
##    ind=[3, 1, 2, 0]
#    plt.subplot(4,1,1+i)
#    CAM_on_input(Mymodel, -2, int(Ytrain[40*i]), X_train_transformed[40*i], -6, 
#                 backprop_modifier = 'relu', grad_modifier= None)
##    CAM_distr(Mymodel, -2, int(Ytrain[40*i]), X_train_transformed[40*i], -6)
#    plt.title('sample {}, True label {}, Pred label{}'.format(40*i, Ytrain[40*i], np.argmax(pred_train[40*i],-1)))
#    


from vis.visualization import visualize_activation
from scipy.signal import periodogram

if Params['exp num'] == 1:
    label_list = ['Moving Left', 'Moving Right', 'Moving away', 'Moving closer']
    seed_list = [0,40,80,120] # for the BCI dataset IV
    fig, ax= plt.subplots(Params['n classes'],2)
    fig_1, ax_1= plt.subplots(Params['n classes'],1)
    
    for i in range(Params['n classes']):
    #    plt.subplot(Params['n classes'],1,1+i)
        _min = np.amin(X_train_transformed[seed_list[i]])
        _max = np.amax(X_train_transformed[seed_list[i]])
        a=visualize_activation(Mymodel, -2, i, seed_input=X_train_transformed[seed_list[i]], 
                              backprop_modifier='relu', input_range=(_min,_max),
                              input_modifiers=None,
                              #max_iter = 1000
                              )
        ax[i][0].plot( a )
        ax[i][0].yaxis.grid(True)
        ax[i][0].set_ylabel(label_list[i])
        ax[i][1].plot( a[:,3] , 'r')
        ax[i][1].yaxis.grid(True)
        ax[i][1].set_yticklabels([])
        
        f_S1, p_S1=periodogram(a, fs=Params['sampling freq'], window=None, nfft=None, 
                  detrend='constant', return_onesided=True, scaling='density', 
                  axis=0)
        ax_1[i].plot(f_S1, p_S1)
        ax_1[i].xaxis.grid(True)
        if i < Params['n classes']-1:
            ax_1[i].set_xticklabels([])
        else:
            ax_1[i].set_xlabel('Frequency(Hz)')
        
        ax_1[i].set_xlim(0,25)
        ax_1[i].set_ylabel('$V^2$/Hz \n  {}'.format(label_list[i]))
    
        
    
    fig.legend(["channel {}".format(i) for i in range(10)], loc='center left', 
                borderpad=1.5, labelspacing=1.5, fontsize = 'xx-large')
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    
    #ax_1[0].legend(['channel {}'.format(i) for i in range(9)])
    fig_1.legend(['channel {}'.format(i) for i in range(10)],loc='center right', ncol = 2)
    fig_1.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
      
elif Params['exp num'] == 2:
    seed_list = [np.where(Ytrain==i)[0][0] for i in range(3)]
    label_list = ['Healthy', 'Preictal', 'Seizure']
    fig, ax= plt.subplots(Params['n classes'],1)
    fig_1, ax_1= plt.subplots(Params['n classes'],1)
    
    for i in range(Params['n classes']):
      #    plt.subplot(Params['n classes'],1,1+i)
          _min = np.amin(X_train_transformed[seed_list[i]])
          _max = np.amax(X_train_transformed[seed_list[i]])
          a=visualize_activation(Mymodel, -2, i, seed_input=X_train_transformed[seed_list[i]], 
                                backprop_modifier='relu', input_range=(_min,_max),
                                input_modifiers=None, max_iter = 1000)
      #    ax[i].plot( (a[:,3]-np.min(a[:,3]))/(np.max(a[:,3])-np.min(a[:,3])) , 'r')
          ax[i].plot(np.linspace(0,23.6,4097),a[:,0] )
          ax[i].yaxis.grid(True)
          ax[i].set_ylabel('{}'.format(label_list[i]))
          
          f_S1, p_S1=periodogram(a, fs=Params['sampling freq'], window=None, nfft=None, 
                                detrend='constant', return_onesided=True, scaling='density', 
                                axis=0)
          ax_1[i].plot(f_S1, p_S1)
          ax_1[i].xaxis.grid(True)
          
          if i < Params['n classes']-1:
            ax_1[i].set_xticklabels([])
            ax[i].set_xticklabels([])
          else:
            ax_1[i].set_xlabel('Frequency(Hz)')
            ax[i].set_xlabel('Time(s)')
            
          ax_1[i].set_xlim(0,40)
          ax_1[i].set_ylabel('$V^2$/Hz \n  {}'.format(label_list[i]))
    
    fig.tight_layout()
    fig_1.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0) 
    plt.subplots_adjust(wspace=0, hspace=0)
    fig, ax= plt.subplots(Params['n classes'],1)
    fig_1, ax_1= plt.subplots(Params['n classes'],1)
    
  
    
else:
    seed_list = [np.where(Ytrain==i)[0][0] for i in range(3)]
    label_list = ['LHT', 'RHT', 'FT']
    fig, ax= plt.subplots(Params['n classes'],1)
    
    for i in range(Params['n classes']):
        _min = np.amin(X_train_transformed[seed_list[i]])
        _max = np.amax(X_train_transformed[seed_list[i]])
        a=visualize_activation(Mymodel, -2, i, seed_input=X_train_transformed[seed_list[i]], 
                              backprop_modifier='relu', input_range=(_min,_max),
                              input_modifiers=None,
                              #max_iter = 1000
                              )
        ax[i].plot( a )
        ax[i].yaxis.grid(True)
        ax[i].set_ylabel(label_list[i])
       
    
    fig.legend(["channel {}".format(i) for i in range(40)], loc='center right', ncol = 4,
                borderpad=1.5, labelspacing=1.5, fontsize = 'xx-large')
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)



# ==================== For the Bonn set ===================================================
# =============================================================================
# =============================================================================
# fig, ax= plt.subplots(Params['n classes'],1)
# fig_1, ax_1= plt.subplots(Params['n classes'],1)
# 
# for i in range(Params['n classes']):
#  #    plt.subplot(Params['n classes'],1,1+i)
#      _min = np.amin(X_train_transformed[seed_list[i]])
#      _max = np.amax(X_train_transformed[seed_list[i]])
#      a=visualize_activation(Mymodel, -2, i, seed_input=X_train_transformed[seed_list[i]], 
#                             backprop_modifier='relu', input_range=(_min,_max),
#                             input_modifiers=None, max_iter = 1000)
#  #    ax[i].plot( (a[:,3]-np.min(a[:,3]))/(np.max(a[:,3])-np.min(a[:,3])) , 'r')
#      ax[i].plot(np.linspace(0,23.6,4097),a[:,0] )
#      ax[i].yaxis.grid(True)
#      ax[i].set_ylabel('{}'.format(label_list[i]))
#      
#      f_S1, p_S1=periodogram(a, fs=Params['sampling freq'], window=None, nfft=None, 
#                             detrend='constant', return_onesided=True, scaling='density', 
#                             axis=0)
#      ax_1[i].plot(f_S1, p_S1)
#      ax_1[i].xaxis.grid(True)
#      
#      if i < Params['n classes']-1:
#         ax_1[i].set_xticklabels([])
#         ax[i].set_xticklabels([])
#      else:
#         ax_1[i].set_xlabel('Frequency(Hz)')
#         ax[i].set_xlabel('Time(s)')
#         
#      ax_1[i].set_xlim(0,40)
#      ax_1[i].set_ylabel('$V^2$/Hz \n  {}'.format(label_list[i]))
# 
# fig.tight_layout()
# fig_1.tight_layout()
# plt.subplots_adjust(wspace=0, hspace=0) 
# plt.subplots_adjust(wspace=0, hspace=0)    
# =============================================================================
# =============================================================================



# ======================For the BCI-3 set=======================================================
# fig, ax= plt.subplots(Params['n classes'],2)
# fig_1, ax_1= plt.subplots(Params['n classes'],1)

# for i in range(Params['n classes']):
# #    plt.subplot(Params['n classes'],1,1+i)
#     _min = np.amin(X_train_transformed[seed_list[i]])
#     _max = np.amax(X_train_transformed[seed_list[i]])
#     a=visualize_activation(Mymodel, -2, i, seed_input=X_train_transformed[seed_list[i]], 
#                           backprop_modifier='relu', input_range=(_min,_max),
#                           input_modifiers=None,
#                           #max_iter = 1000
#                           )
#     ax[i][0].plot( a )
#     ax[i][0].yaxis.grid(True)
#     ax[i][0].set_ylabel(label_list[i])
#     ax[i][1].plot( a[:,3] , 'r')
#     ax[i][1].yaxis.grid(True)
#     ax[i][1].set_yticklabels([])
    
#     f_S1, p_S1=periodogram(a, fs=Params['sampling freq'], window=None, nfft=None, 
#               detrend='constant', return_onesided=True, scaling='density', 
#               axis=0)
#     ax_1[i].plot(f_S1, p_S1)
#     ax_1[i].xaxis.grid(True)
#     if i < Params['n classes']-1:
#         ax_1[i].set_xticklabels([])
#     else:
#         ax_1[i].set_xlabel('Frequency(Hz)')
    
#     ax_1[i].set_xlim(0,25)
#     ax_1[i].set_ylabel('$V^2$/Hz \n  {}'.format(label_list[i]))

    

# fig.legend(["channel {}".format(i) for i in range(10)], loc='center left', 
#             borderpad=1.5, labelspacing=1.5, fontsize = 'xx-large')
# fig.tight_layout()
# plt.subplots_adjust(wspace=0, hspace=0)

# #ax_1[0].legend(['channel {}'.format(i) for i in range(9)])
# fig_1.legend(['channel {}'.format(i) for i in range(10)],loc='center right', ncol = 2)
# fig_1.tight_layout()
# plt.subplots_adjust(wspace=0, hspace=0)

# =============================================================================
