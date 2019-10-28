# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:40:42 2019

@author: dykua

A test for the dynamic connectivity matrix + Dense + DNN idea
"""

from keras.optimizers import Adam, SGD
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

Params = {
        'batchsize': 32,
        'epochs': 100,
        'lr': 1e-4,
        'cut_off freq': 0.1,
        'win len': 10,
        'skip': 5
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
from preprocess import normalize_samples, Buterworth_batch, connect_matrix, get_lower_info

#X_train_transformed = Buterworth_batch(Xtrain, cut_off_freq = Params['cut_off freq'])
#X_test_transformed = Buterworth_batch(Xtest, cut_off_freq = Params['cut_off freq'])
#
#X_train_transformed, X_test_transformed, _ = normalize_samples(X_train_transformed, X_test_transformed, MinMaxScaler, 0, 1)

X_train_DCM = connect_matrix(Xtrain, Params['win len'], Params['skip'])
X_train_OF = get_lower_info(X_train_DCM)
X_test_DCM = connect_matrix(Xtest, Params['win len'], Params['skip'])
X_test_OF = get_lower_info(X_test_DCM)

#Params['samples'], Params['t-length'], Params['feature dim'] = X_train_transformed.shape

Params['samples'], Params['num frames'], Params['feature dim'] = X_train_OF.shape

X_train_transformed, X_test_transformed, _ = normalize_samples(X_train_OF, X_test_OF, MinMaxScaler, 0, 1)

'''
One-Hot labels
'''
enc = OneHotEncoder()
Ytrain_OH = enc.fit_transform(Ytrain[:,None])
Ytest_OH = enc.transform(Ytest[:,None])
_, Params['n classes'] = Ytrain_OH.shape


'''
Build the feature extractor
'''
from keras.layers import Dense, Input, TimeDistributed, SimpleRNN, Flatten 
from keras.models import Model
from keras.optimizers import Adam

def feature_extractor(steps, fdim):
    '''
    Each time step will have the same dense layer
    '''
    x_in = Input(shape=(steps, fdim))
    
#    x = TimeDistributed(Dense(128, activation='elu'))(x_in)
    x = TimeDistributed(Dense(64, activation='elu'))(x_in)
    x_out = TimeDistributed(Dense(32, activation='tanh'))(x)
    
    encoder = Model(x_in, x_out, name = 'extractor')
    
    return encoder



'''
Build the RNN
'''

def clf_RNN(steps, fdim):
    '''
    It takes the time dependent feature and spits out the probability 
    '''
    x_in = Input(shape=(steps, fdim))
    x    = SimpleRNN(64, activation='tanh', use_bias=False, 
                      kernel_initializer='glorot_uniform', 
                      recurrent_initializer='orthogonal', 
                      bias_initializer='zeros', kernel_regularizer=None, 
                      recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
                      kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, 
                      dropout=0.3, recurrent_dropout=0.0, return_sequences=True, return_state=False, 
                      go_backwards=False, stateful=False, unroll=False)(x_in)
    
    x    = TimeDistributed(Dense(1, activation='tanh'))(x) 
    
    x_flat = Flatten()(x)
#    x_flat = Dense(32, activation='relu')(x_flat)
    x_out = Dense(4, activation='softmax')(x_flat) 
    clf = Model(x_in, x_out, name = 'classifier') # x_out is of shape 3d
    
    return clf


'''
Build the whole model
'''
def whole_model(encoder, clf, steps, fdim):
    x_in = Input(shape = (steps, fdim))
    feature = encoder(x_in)
    pred = clf(feature)
    model = Model(x_in, pred)
    
    return model

Mymodel = whole_model(feature_extractor(Params['num frames'], Params['feature dim']), 
                      clf_RNN(Params['num frames'], 32), 
                      Params['num frames'], Params['feature dim'])
Mymodel.summary()
Mymodel.compile(loss='categorical_crossentropy', 
                    metrics=['accuracy'],
                    optimizer=Adam(lr=Params['lr']))
hist = Mymodel.fit(X_train_transformed, Ytrain_OH, 
                epochs=Params['epochs'], batch_size = Params['batchsize'],
                validation_data = (X_test_transformed, Ytest_OH),
    #            validation_split=0.2,
                verbose=1)

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
    

