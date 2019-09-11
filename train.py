# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 15:45:05 2019

@author: dykua

Training script

Q:  30% gap accuracy between train and test
      More data --- data augmentation?
      A simpler Model? ---
      
      
ToDo: Try the rolling mean, more regularization.
"""

from keras.optimizers import Adam, SGD
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

Params = {
        'batchsize': 32,
        'epochs': 100,
        'lr': 1e-4,
        'cut_off freq': 0.1,
        'num downsampling': None
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
Xtrain = np.load(r'../Xtrain.npy')
Xtest = np.load(r'../Xtest.npy')
Ytrain = np.load(r'../Ytrain1.npy')
Ytest = np.load(r'../Ytest1.npy')
'''
Normalize data
'''

#X_train_transformed = Xtrain[:,:75,:]
#X_test_transformed = Xtest[:,:75,:]
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from preprocess import normalize_in_time, normalize_samples, rolling_max, normalize_mvar, Buterworth_batch

X_train_transformed = Buterworth_batch(Xtrain, cut_off_freq = Params['cut_off freq'])
X_test_transformed = Buterworth_batch(Xtest, cut_off_freq = Params['cut_off freq'])

X_train_transformed, X_test_transformed, _ = normalize_samples(X_train_transformed, X_test_transformed, MinMaxScaler, 0, 1)
#X_train_transformed, X_test_transformed, _ = normalize_samples(Xtrain, Xtest, MinMaxScaler, 0, 1)

#X_train_transformed, X_test_transformed = normalize_mvar(X_train_transformed, X_test_transformed)

#X_train_transformed = normalize_in_time(Xtrain,smooth = True, cut_off_freq = Params['cut_off freq'], order = 3)
#X_test_transformed = normalize_in_time(Xtest,smooth = True, cut_off_freq = Params['cut_off freq'], order = 3)
#X_train_transformed, X_test_transformed, _ = normalize_samples(X_train_transformed, X_test_transformed, MinMaxScaler, 0, 1)

#X_train_transformed = X_train_transformed[:,:,:2]
#X_test_transformed = X_test_transformed[:,:,:2]

#X_train_transformed = rolling_mean(X_train_transformed, 30)
#X_test_transformed = rolling_mean(X_test_transformed, 30)

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
                      dropoutRate = 0.3, kernLength = 50, F1 = 32, 
                      D = 2, F2 = 64, norm_rate = 0.25, 
                      optimizer = Adam,
                      learning_rate=Params['lr'],
                      dropoutType = 'Dropout',
                      act = 'softmax')

Mymodel.summary()


from keras.callbacks import TensorBoard, ReduceLROnPlateau, LearningRateScheduler
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

'''
Train Model
'''
randshift = False

if randshift:
    from Utils import generator
    gen = generator(X_train_transformed, Ytrain_OH, Params['batchsize'], shift_limit = 40)
    hist = Mymodel.fit_generator(gen, steps_per_epoch=160//Params['batchsize'] + 1, 
                                 epochs=Params['epochs'], 
                                 verbose=1, callbacks=None, 
                                 validation_data=(X_test_transformed, Ytest_OH), 
                                 validation_steps=None, 
                                 class_weight=None, max_queue_size=10, 
                                 workers=1, use_multiprocessing=False, shuffle=True)
    
else:
    weight_dict = np.array([1.0,10.0,1.0,1.0])
    hist = Mymodel.fit(X_train_transformed, Ytrain_OH, 
                epochs=Params['epochs'], batch_size = Params['batchsize'],
                validation_data = (X_test_transformed, Ytest_OH),
    #            validation_split=0.2,
                verbose=1,
                callbacks=[],
                class_weight = weight_dict)

#Mymodel.save_weights('Mymodel_weights.h5')

'''
Visualize training history
'''
import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.figure()
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

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
import keras.backend as K
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
    
'''
Forming a combined classifier: NN + SVM
'''
from sklearn import svm
clf = svm.SVC(C=1.0, gamma=0.001, decision_function_shape='ovr', 
              kernel = 'rbf', degree=3,
              class_weight=None, tol=1e-3)

act = K.function([Mymodel.layers[0].input], [Mymodel.get_layer('dense').output])

feature = act([X_train_transformed])
feature_test = act([X_test_transformed])
to_svm = feature[-1]
to_svm_test = feature_test[-1]

clf.fit(to_svm, Ytrain)
print("Acc with an extra SVM: {}".format(accuracy_score(clf.predict(to_svm_test), Ytest)))

from visual import CAM_on_input
plt.figure()
for i in range(4):
#    ind=[3, 1, 2, 0]
    plt.subplot(4,1,1+i)
    CAM_on_input(Mymodel, -2, Ytrain[40*i], X_train_transformed[40*i], -5)
    plt.title('sample {}, True label {}, Pred label{}'.format(40*i, Ytrain[40*i], np.argmax(pred_train[40*i],-1)))
    
