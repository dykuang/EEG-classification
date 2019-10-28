# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 18:25:34 2019

@author: dykua

test the attention layer's location

Q: attention is done channel wise or share cross different channels?
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


dataset = 'D:/EEG/archive/BCI-IV-dataset3/'
subject = 2
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

X_train_transformed, X_test_transformed, _ = normalize_samples(X_train_transformed, X_test_transformed, MinMaxScaler, 0, 1)


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
from architectures import My_eeg_net_1d_w_att as eeg_net
    

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

from visual import CAM_on_input
plt.figure()
for i in range(4):
#    ind=[3, 1, 2, 0]
    plt.subplot(4,1,1+i)
    CAM_on_input(Mymodel, -2, Ytrain[40*i], X_train_transformed[40*i], -5)
    plt.title('sample {}, True label {}, Pred label{}'.format(40*i, Ytrain[40*i], np.argmax(pred_train[40*i],-1)))
    

