#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 16:23:27 2019

@author: dykuang

Test script with added modules
"""

from keras.optimizers import Adam, SGD
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

Params = {
        'batchsize': 32,
        'epochs': 200,
        'lr': 1e-4,
        'cut_off freq': 0.1,
        'LW': [1.0, 1e-4],
        'share': True,
        'Sub ratio': 2
        }

'''
Load data
'''
#Xtrain = np.load(r'../MI_train.npy')[:,:1000,:]
#Xtest = np.load(r'../MI_test.npy')[:,:1000,:]
#Ytrain = np.load(r'../MI_train_D1_label.npy')
#Ytest = np.load(r'../MI_test_D1_label.npy')
#
dataset = 'D:/EEG/archive/BCI-IV-dataset3/'
subject = 1
Xtrain = np.load(dataset+r'S{}train.npy'.format(subject))
Xtest = np.load(dataset+r'S{}test.npy'.format(subject))
Ytrain = np.load(dataset+r'Ytrain.npy'.format(subject))
Ytest = np.load(dataset+r'S{}Ytest.npy'.format(subject))[0]

'''
Normalize data
'''

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from preprocess import normalize_in_time, normalize_samples, rolling_max, normalize_mvar, Buterworth_batch

X_train_transformed = Buterworth_batch(Xtrain, cut_off_freq = Params['cut_off freq'])
X_test_transformed = Buterworth_batch(Xtest, cut_off_freq = Params['cut_off freq'])

#X_train_transformed, X_test_transformed, _ = normalize_samples(X_train_transformed, X_test_transformed, MinMaxScaler, 0, 1)

X_train_transformed, X_test_transformed, _ = normalize_samples(Xtrain, Xtest, MinMaxScaler, 0, 1)
#X_train_transformed, X_test_transformed = normalize_mvar(X_train_transformed, X_test_transformed)

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

from architectures import My_eeg_net_1d_resample as eeg_net
from architectures import My_eeg_net_1d, locnet, loc_Unet
from keras.optimizers import Adam

if Params['share']:
    disp_net = locnet(Samples = Params['t-length'], 
                      Chans = Params['feature dim'], 
                      kernLength = 5,
                      output_channels = 1,
                      pooling = Params['Sub ratio'])
else:
    disp_net = locnet(Samples = Params['t-length'], 
                      Chans = Params['feature dim'], 
                      kernLength = 5,
                      output_channels = Params['feature dim'],
                      pooling = Params['Sub ratio'])
    
disp_net.name = "Resampler"

cl_net = My_eeg_net_1d(Params['n classes'], Chans = Params['feature dim'], 
                  Samples = Params['t-length']//Params['Sub ratio'], 
                  dropoutRate = 0.3, kernLength = 50, F1 = 32, 
                  D = 2, F2 = 64, norm_rate = 0.25, 
                  optimizer = Adam,
                  learning_rate=Params['lr'],
                  dropoutType = 'Dropout')
cl_net.name = "Classifier"


Mymodel = eeg_net(Sampler=disp_net, Classifier=cl_net, 
                  t_length=Params['t-length'], Chans=Params['feature dim'],
                  optimizer=Adam(lr=Params['lr']), 
                  loss_weights = Params['LW'],
                  share = Params['share'],                  
                  pooling = Params['Sub ratio'])

Mymodel.summary()

#from datetime import datetime
#disp_net.save_weights('Resampler_Weights_'+datetime.now().strftime("%Y%m%d-%H%M%S") + '.h5')
#cl_net.save_weights('Classifier_Weights_'+datetime.now().strftime("%Y%m%d-%H%M%S") + '.h5')


from keras.callbacks import TensorBoard, ReduceLROnPlateau, LearningRateScheduler
from datetime import datetime

'''
Some call back functions
'''
logdir = "tf-log-test/"+datetime.now().strftime("%Y%m%d-%H%M%S")+'/'
tf_callback = TensorBoard(log_dir = logdir, 
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True,
                          write_grads=False,
                          batch_size = 32,
                          embeddings_freq=0, 
                          embeddings_layer_names=None, 
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

'''
Train Model
'''
hist = Mymodel.fit(X_train_transformed, [Ytrain_OH, np.zeros([Params['samples'], Params['t-length']//Params['Sub ratio'], Params['feature dim']])], 
            epochs=Params['epochs'], batch_size = Params['batchsize'],
            validation_data = (X_test_transformed, [Ytest_OH, np.zeros([X_test_transformed.shape[0], X_test_transformed.shape[1]//Params['Sub ratio'], Params['feature dim']])]),
#            validation_split=0.2,
            verbose=1,
            callbacks=[])

#Mymodel.save_weights('Mymodel_weights_'+datetime.now().strftime("%Y%m%d-%H%M%S")+'.h5')

'''
Summary statistics
'''
print(Params)
from sklearn.metrics import accuracy_score
pred_train,_ = Mymodel.predict(X_train_transformed)
print("Acc on trained data: {}".format(accuracy_score(Ytrain, np.argmax(pred_train, axis=1))))
pred_test,_ = Mymodel.predict(X_test_transformed)
print("Acc on test data: {}".format(accuracy_score(Ytest, np.argmax(pred_test, axis=1))))

'''
For test purpose
'''
import keras.backend as K
re = K.function([Mymodel.input], [Mymodel.layers[-3].output])
a = re([X_train_transformed])[0]

disp = Mymodel.layers[-3].locnet.predict(X_test_transformed)
grid = np.linspace(0, Params['t-length']-1, Params['t-length']//Params['Sub ratio'])
grid = np.tile(grid, Xtest.shape[0]*Xtest.shape[-1])
grid = np.reshape(grid, (Xtest.shape[0], Xtest.shape[-1], Xtest.shape[1]//Params['Sub ratio']))
grid = np.transpose(grid, (0, 2, 1))
moved = grid + disp
#moved = grid+disp[0,:,0]


import matplotlib.pyplot as plt
plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss')
plt.legend(['Train', 'Test'])

plt.figure()
plt.plot(hist.history['Classifier_acc'])
plt.plot(hist.history['val_Classifier_acc'])
plt.title('Accuracy')
plt.legend(['Train', 'Test'])

plt.figure()
plt.subplot(2,1,1)
plt.stem(disp[0,:,0])
plt.subplot(2,1,2)
plt.hist(disp[0,:,0])

plt.figure()
plt.plot(np.arange(Params['t-length']), X_train_transformed[0,:,0])
#plt.plot(moved[0,:,0], a[0,:,0])
plt.plot(np.arange(0, Params['t-length'],Params['Sub ratio']), a[0,:,0])
plt.legend(['original','resampled'])



#plt.figure()
#plt.scatter(moved, np.zeros_like(grid))
#plt.figure()
#plt.hist(moved)
#plt.figure()
#plt.plot(moved)
