# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 15:02:50 2019

@author: dykua

test the classifier with the static connectivity matrix as the extra input
"""

from keras.optimizers import Adam, SGD
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

Params = {
        'batchsize': 32,
        'epochs': 40,
        'lr': 1e-4,
        'cut_off freq': 0.1,
        'num downsampling': None,
        'low cut': 0.01,
        'high cut': 199.99/2,
        'sampling freq': 400,  # 173.61 for Bonn, 400 for BCI-dataset 3
        'CM win len': 10,
        'CM win skip': 10
        }



'''
Load data
'''

# ============= BCI-IV-3 ==============================================
# =============================================================================
# dataset = 'D:/EEG/archive/BCI-IV-dataset3/'
# subject = 1
# Xtrain = np.load(dataset+r'S{}train.npy'.format(subject))
# Xtest = np.load(dataset+r'S{}test.npy'.format(subject))
# Ytrain = np.load(dataset+r'Ytrain.npy'.format(subject))
# Ytest = np.load(dataset+r'S{}Ytest.npy'.format(subject))[0]
# =============================================================================

# ======================== fNIRs ==============================================d
from scipy.io import loadmat
dataset = 'C:/Users/dykua/matlab projects/BCI/'


subject = 5
X = loadmat(dataset + 'x{:02d}_m99_nobc.mat'.format(subject))['x'].transpose((2,0,1))[...,::2]
Y = loadmat(dataset + 'label{:02d}.mat'.format(subject))['y'].flatten()-1

# X = []
# Y = []
# for subject in range(1,31):
#     X.append(loadmat(dataset + 'x{:02d}_m99.mat'.format(subject))['x'].transpose((2,0,1)))
#     Y.append(loadmat(dataset + 'label{:02d}.mat'.format(subject))['y'].flatten()-1 )
# X = np.vstack(X)
# Y = np.hstack(Y)

# from sklearn.utils import shuffle
# X, Y = shuffle(X, Y)

from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(
                                X, Y, test_size=0.20, random_state=42,
                                shuffle = True, stratify = Y
                                )


'''
Normalize data
'''

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from preprocess import normalize_in_time, normalize_samples, normalize_mvar, Buterworth_batch, connect_matrix, get_lower_info, batch_band_pass
#
#X_train_transformed = Buterworth_batch(Xtrain, cut_off_freq = Params['cut_off freq'])
#X_test_transformed = Buterworth_batch(Xtest, cut_off_freq = Params['cut_off freq'])
##
# X_train_transformed, X_test_transformed, _ = normalize_samples(Xtrain, Xtest, StandardScaler, 0, 1)

# Xtrain =  batch_band_pass(Xtrain, Params['low cut'], Params['high cut'], Params['sampling freq'])
# Xtest =  batch_band_pass(Xtest, Params['low cut'], Params['high cut'], Params['sampling freq'])

from scipy.stats import zscore
X_train_transformed = zscore(Xtrain, axis=1)
X_test_transformed = zscore(Xtest, axis=1)

# X_train_transformed = zscore(X_train_transformed, axis=1)
# X_test_transformed = zscore(X_test_transformed, axis=1)

Params['samples'], Params['t-length'], Params['feature dim'] = X_train_transformed.shape
Params['CM shape'] = ( (Params['t-length'] - Params['CM win len'])//Params['CM win skip'] + 1, 
                      (Params['feature dim']**2 - Params['feature dim'])//2 )
#X_train_CM = connect_matrix(X_train_transformed, 400, 1)[:,0,...,None]
#X_test_CM = connect_matrix(X_test_transformed, 400, 1)[:,0,...,None]

#
#X_train_CM = connect_matrix(X_train_transformed, 50, 25).transpose((0,2,3,1))
#X_test_CM = connect_matrix(X_test_transformed, 50, 25).transpose((0,2,3,1))

X_train_CM = connect_matrix(Xtrain , Params['CM win len'], Params['CM win skip'])
X_test_CM = connect_matrix(Xtest, Params['CM win len'], Params['CM win skip'])

X_train_CM = get_lower_info(X_train_CM)
X_test_CM = get_lower_info(X_test_CM)

X_train_CM = zscore(X_train_CM, axis=1)
X_test_CM = zscore(X_test_CM, axis=1)

#
#X_train_CM = (X_train_CM - np.amin(X_train_CM))/2
#X_test_CM = (X_test_CM - np.amin(X_test_CM))/2

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
from architectures import My_eeg_net_1d_w_CM as eeg_net
    

Mymodel = eeg_net(Params['n classes'], Chans = Params['feature dim'], 
                      Samples = Params['t-length'], 
                      dropoutRate = 0.5, kernLength = 50, F1 = 32, 
                      D = 2, F2 = 64, norm_rate = 0.25, 
                      optimizer = Adam,
                      learning_rate=Params['lr'],
                      dropoutType = 'Dropout',
                      act = 'softmax',
                      CM_shape = Params['CM shape'] )

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
randfreq = True
aug = False
if aug:
    if randshift:
        from Utils import generator
        gen = generator(X_train_transformed, Ytrain_OH, Params['batchsize'], shift_limit = 40)
        hist = Mymodel.fit_generator(gen, steps_per_epoch=Params['samples']//Params['batchsize'] + 1, 
                                     epochs=Params['epochs'], 
                                     verbose=1, callbacks=None, 
                                     validation_data=(X_test_transformed, Ytest_OH), 
                                     validation_steps=None, 
                                     class_weight=None, max_queue_size=10, 
                                     workers=1, use_multiprocessing=False, shuffle=True)
    if randfreq:
        from Utils import generator_freq
        gen = generator_freq(X_train_transformed, Ytrain_OH, Params['batchsize'], bandwidth = 50, dist_to_end = 100)
        hist = Mymodel.fit_generator(gen, steps_per_epoch=Params['samples']//Params['batchsize'] + 1, 
                                     epochs=Params['epochs'], 
                                     verbose=1, callbacks=None, 
                                     validation_data=(X_test_transformed, Ytest_OH), 
                                     validation_steps=None, 
                                     class_weight=None, max_queue_size=10, 
                                     workers=1, use_multiprocessing=False, shuffle=True)

else:
    weight_dict = np.array([1.0,1.0,1.0,1.0])
    hist = Mymodel.fit([X_train_transformed, X_train_CM], Ytrain_OH, 
                epochs=Params['epochs'], batch_size = Params['batchsize'],
                validation_data = ([X_test_transformed, X_test_CM], Ytest_OH),
                # validation_split=0.2,
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
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])

'''
Summary statistics
'''
print(Params)
from sklearn.metrics import accuracy_score
pred_train = Mymodel.predict([X_train_transformed, X_train_CM])
print("Acc on trained data: {}".format(accuracy_score(Ytrain, np.argmax(pred_train, axis=1))))
pred_test = Mymodel.predict([X_test_transformed, X_test_CM])
print("Acc on test data: {}".format(accuracy_score(Ytest, np.argmax(pred_test, axis=1))))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(Ytest, np.argmax(pred_test, axis=1)) )


'''
keras-vis does not seem to support multiple input yet
'''
# from vis.visualization import visualize_activation

# seed_list = [np.where(Ytrain==i)[0][0] for i in range(3)]
# label_list = ['LHT', 'RHT', 'FT']
# fig, ax= plt.subplots(Params['n classes'],1)

# for i in range(Params['n classes']):
#     _min = np.amin(X_train_transformed[seed_list[i]])
#     _max = np.amax(X_train_transformed[seed_list[i]])
#     a=visualize_activation(Mymodel, -2, i, seed_input=[X_train_transformed[seed_list[i]], X_train_CM[seed_list[i]]], 
#                           backprop_modifier='relu', input_range=(_min,_max),
#                           wrt_tensor = Mymodel.layers[0],
#                           input_modifiers=None,
#                           #max_iter = 1000
#                           )
#     ax[i][0].plot( a )
#     ax[i][0].yaxis.grid(True)
#     ax[i][0].set_ylabel(label_list[i])
#     ax[i][1].plot( a[:,3] , 'r')
#     ax[i][1].yaxis.grid(True)
#     ax[i][1].set_yticklabels([])
       

# fig.legend(["channel {}".format(i) for i in range(10)], loc='center left', 
#             borderpad=1.5, labelspacing=1.5, fontsize = 'xx-large')
# fig.tight_layout()
# plt.subplots_adjust(wspace=0, hspace=0)

    

    
