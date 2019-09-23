# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 13:48:44 2019

@author: dykua

Test the window module
"""


from keras.optimizers import Adam, SGD
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

Params = {
        'batchsize': 32,
        'epochs': 100,
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


'''
Build Model
'''

from architectures import My_eeg_net_window as eeg_net
from architectures import My_eeg_net_1d, locnet_window
from keras.optimizers import Adam

win_net = locnet_window(Samples = Params['t-length'], 
                        Chans= Params['feature dim'], 
                        kernLength = 30, 
                        share_w_channels = Params['share_w_channels'])

win_net.name = "Window"

cl_net = My_eeg_net_1d(Params['n classes'], Chans = Params['feature dim'], 
                  Samples = Params['win len'], 
                  dropoutRate = 0.2, kernLength = 50, F1 = 32, 
                  D = 2, F2 = 64, norm_rate = 0.25, 
                  optimizer = Adam,
                  learning_rate=Params['lr'],
                  dropoutType = 'Dropout')
cl_net.name = "Classifier"


Mymodel = eeg_net(Window = win_net, Classifier=cl_net,
                  t_length = Params['t-length'], 
                  window_len = Params['win len'],
                  Chans = Params['feature dim'],
                  optimizer=Adam(lr=Params['lr']),
                  share_w_channels = Params['share_w_channels']
                  )

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

transformation_pre = win_net.predict(X_train_transformed)

hist = Mymodel.fit(X_train_transformed, Ytrain_OH, 
            epochs=Params['epochs'], batch_size = Params['batchsize'],
            validation_data = (X_test_transformed, Ytest_OH),
#            validation_split=0.2,
            verbose=1,
            callbacks=[])

#Mymodel.save_weights('Mymodel_weights_'+datetime.now().strftime("%Y%m%d-%H%M%S")+'.h5')

'''
Summary statistics
'''
print(Params)
from sklearn.metrics import accuracy_score
pred_train = Mymodel.predict(X_train_transformed)
print("Acc on trained data: {}".format(accuracy_score(Ytrain, np.argmax(pred_train, axis=1))))
pred_test = Mymodel.predict(X_test_transformed)
print("Acc on test data: {}".format(accuracy_score(Ytest, np.argmax(pred_test, axis=1))))

'''
For test purpose
'''
import keras.backend as K
re = K.function([Mymodel.input], [Mymodel.layers[-2].output])
a = re([X_train_transformed])[0]

#start_point = Mymodel.layers[-2].locnet.predict(X_test_transformed)
#start_point_scaled_back = start_point * (Params['t-length']-1)
#start_point_scaled_back = np.floor(start_point_scaled_back)

transformation = win_net.predict(X_train_transformed)
#grid = np.arange(Params['win len'])
grid = np.linspace(0.0, 1.0, Params['win len'])
indices_grid = np.stack([grid, np.ones_like(grid)], axis=0).flatten()

share_w_channels = Params['share_w_channels']
if share_w_channels:
    indices_grid = np.tile(indices_grid, np.stack([Params['samples']]))
    indices_grid = np.reshape(indices_grid, (Params['samples'], 2, -1) )
    
    #transformation = np.stack([0.5*np.ones( (Params['samples'], 1) ), 
    #                           100*np.ones( (Params['samples'], 1) )], axis= 1)
    
    transformation = np.reshape(transformation, (-1, 1, 2))
    
    grid_transformed = np.matmul(transformation, indices_grid)
else:
    indices_grid = np.tile(indices_grid, np.stack([Params['samples']*Params['feature dim']]))
    indices_grid = np.reshape(indices_grid, (Params['samples'], Params['feature dim'], 2, -1) )
    
    transformation = np.reshape(transformation, (-1, Params['feature dim'], 1, 2))
    
    grid_transformed = np.matmul(transformation, indices_grid)

grid_transformed = grid_transformed * (Params['win len'] - 1)




import matplotlib.pyplot as plt
plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss')
plt.legend(['Train', 'Test'])

plt.figure()
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Accuracy')
plt.legend(['Train', 'Test'])

#plt.figure()
#plt.hist(start_point_scaled_back[:,5])

if share_w_channels:
    plt.figure()
    plt.subplot(2,2,1)
    plt.hist(transformation_pre[:,0])
    plt.subplot(2,2,2)
    plt.hist(transformation_pre[:,1])
    plt.subplot(2,2,3)
    plt.hist(transformation[:,0,0])
    plt.subplot(2,2,4)
    plt.hist(transformation[:,0,1])


ind =25
channel = 0
plt.figure()
plt.plot(np.arange(Params['t-length']), X_train_transformed[ind,:,channel])
if share_w_channels:
    plt.plot(grid_transformed[ind,channel,:], a[ind,:,channel])
else:
    plt.plot(grid_transformed[ind,channel,0,:], a[ind,:,channel])
plt.legend(['original','resampled'])



#plt.figure()
#plt.scatter(moved, np.zeros_like(grid))
#plt.figure()
#plt.hist(moved)
#plt.figure()
#plt.plot(moved)
