# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 20:27:51 2019

@author: dykua

deploy modules at different locations of the classfication network 
"""

from keras.optimizers import Adam, SGD
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

Params = {
        'batchsize': 32,
        'epochs': 60,
        'lr': 1e-4,
        'cut_off freq': 0.1,
        'trunc rate': 1.0,
        'share_w_channels': True
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
Params['win len'] = int(np.floor(Params['t-length'] * Params['trunc rate']))

'''
One-Hot labels
'''
enc = OneHotEncoder()
Ytrain_OH = enc.fit_transform(Ytrain[:,None])
Ytest_OH = enc.transform(Ytest[:,None])
_, Params['n classes'] = Ytrain_OH.shape


from keras.models import Model
from keras.layers import Input, SeparableConv1D, Conv1D, Dense, Flatten, Dropout, BatchNormalization, concatenate, add, MaxPooling1D,PReLU
from keras.optimizers import Adam
from keras.layers import UpSampling1D, MaxPooling1D, GlobalAveragePooling1D, SpatialDropout1D, AveragePooling1D, SeparableConv1D, Activation, AveragePooling2D
from keras.constraints import max_norm
import keras.backend as K

from modules import Window_trunc, Window_trunc_no_weights, SpatialTransformer, STN_1D, STN_1D_noweights, STN_1D_noweights_multi_channel

from architectures import My_eeg_net_1d, locnet_window

Samples = 400
Chans = 10
F1 = 32
F2 = 64
kernLength = 50
dropoutRate = 0.2
### pretrain ##################################################################
cl_net = My_eeg_net_1d(Params['n classes'], Chans = Params['feature dim'], 
                  Samples = Params['t-length'], 
                  dropoutRate = dropoutRate, kernLength = kernLength, F1 = F1, 
                  D = 2, F2 = F2, norm_rate = 0.25, 
                  optimizer = Adam,
                  learning_rate=Params['lr'],
                  dropoutType = 'Dropout')

layer_names = [layer.name for layer in cl_net.layers]

prehist = cl_net.fit(X_train_transformed, Ytrain_OH, 
            epochs=Params['epochs'], batch_size = Params['batchsize'],
            validation_data = (X_test_transformed, Ytest_OH),
#            validation_split=0.2,
            verbose=1,
            callbacks=[])




### retrain  ###################################################################

input1   = Input(shape = (Samples, Chans), name='input') # treat multi-channel eeg as one frame of 2d-image

##################################################################
#    block1       = GaussianNoise(0.2)(input1)
block1       = Conv1D(F1, kernLength, padding = 'same',
                                   use_bias = False, name = 'block1-C1D')(input1)  
# block1 = My_LSTM(block1, 32, 1, 32, inter_dim_list=None)
    
block1       = BatchNormalization(name = 'block1-BN1')(block1)
block1       = SeparableConv1D(F2, 8, use_bias = False,  #no Depthwise1D
                                   depthwise_constraint = max_norm(1.),
                                   padding = 'same', name = 'block1-SC1D')(block1)
block1       = BatchNormalization(name = 'block1-BN2')(block1)
block1       = Activation('elu',name='block1-AC')(block1)

########### inserted #####################################################################
win_net = locnet_window(Samples = K.int_shape(block1)[1], 
                        Chans= K.int_shape(block1)[-1], 
                        kernLength = 8, 
                        share_w_channels = Params['share_w_channels'])
trans       = win_net(block1)
windowed_signal = STN_1D_noweights(output_size=K.int_shape(block1)[1:])([trans, block1])
#########################################################################################


#    block1       = AveragePooling1D(8)(block1)
block1       = MaxPooling1D(8, name = 'block1-MP')(windowed_signal)
block1       = Dropout(dropoutRate, name='block1-DP')(block1)
    
block2       = SeparableConv1D(F2, 8, use_bias = False, 
                               padding = 'same', name = 'block2-SC1D')(block1)
block2       = BatchNormalization(name = 'block2-BN')(block2)
block2       = Activation('elu', name = 'block2-AC')(block2)
block2       = AveragePooling1D(4, name = 'block2-AP')(block2)

block2       = Dropout(dropoutRate, name = 'block2-DP')(block2)

           
flatten      = Flatten(name = 'flatten')(block2)

    
dense        = Dense(4, name = 'dense', 
                    kernel_constraint = max_norm(0.25))(flatten)
    

softmax      = Activation('softmax', name = 'softmax')(dense)
    
Mymodel = Model(input1, softmax)


for L_name in layer_names[:]:
    Mymodel.get_layer(L_name).set_weights(cl_net.get_layer(L_name).get_weights())
    Mymodel.get_layer(L_name).trainable = False
    
############# pretrain win_net #####################
Reg_net = Model(input1, trans)
Reg_net.compile(loss='mse',optimizer=Adam(lr=5e-4))
ide = np.stack([np.ones([160,1]), np.zeros([160,1])], axis = 1)[...,0]
Reg_net.fit(X_train_transformed, ide,
            epochs = 50, batch_size = 32, verbose = 1)

import matplotlib.pyplot as plt
transformation = Reg_net.predict(X_train_transformed)
plt.figure()
plt.subplot(1,2,1)
plt.hist(transformation[:,0])
plt.subplot(1,2,2)
plt.hist(transformation[:,1])
#########################################################

Mymodel.compile(loss='categorical_crossentropy', 
                    metrics=['accuracy'],
                    optimizer=Adam(lr=1e-4))

hist = Mymodel.fit(X_train_transformed, Ytrain_OH, 
            epochs=Params['epochs'], batch_size = Params['batchsize'],
            validation_data = (X_test_transformed, Ytest_OH),
#            validation_split=0.2,
            verbose=1,
            callbacks=[])


############# visualization #########################

feature_func = K.function([input1], [Mymodel.get_layer('block1-AC').output, windowed_signal])
features = feature_func([X_train_transformed])

plt.figure()
plt.plot(features[0][0,:,0])
plt.plot(features[1][0,:,0])


transformation = Reg_net.predict(X_train_transformed)
plt.figure()
plt.subplot(1,2,1)
plt.hist(transformation[:,0])
plt.subplot(1,2,2)
plt.hist(transformation[:,1])
