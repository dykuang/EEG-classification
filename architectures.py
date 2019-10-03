# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 15:45:47 2019

@author: dykua

Contains some network architectures
"""


from keras.models import Model
from keras.layers import Input, SeparableConv1D, Conv1D, Dense, Flatten, LeakyReLU, Dropout, BatchNormalization, concatenate, add, MaxPooling1D,PReLU
from keras.optimizers import Adam
from keras.layers import UpSampling1D, MaxPooling1D, GlobalAveragePooling1D, SpatialDropout1D, Conv2D, AveragePooling1D, SeparableConv2D, SpatialDropout2D, DepthwiseConv2D, SeparableConv1D, Activation, AveragePooling2D
from keras.constraints import max_norm

# model

'''
Define a LSTM block
'''
from keras.layers import LSTM, TimeDistributed
def My_LSTM(x, hid_dim, num_layers, out_dim, inter_dim_list=[32, 32], activation_out = 'tanh'):
    if num_layers:
        for i in range(num_layers):
            x = LSTM(hid_dim,return_sequences=True)(x)
    if inter_dim_list is not None:
        for j in inter_dim_list:
            x = TimeDistributed(Dense(j, activation='tanh'))(x)
    x = TimeDistributed(Dense(out_dim, activation=activation_out))(x)

    return x


'''
Some architectures from github.

Q: Do not understand why use conv2D?
'''
def My_eeg_net(nb_classes, Chans = 64, Samples = 128, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, 
             optimizer = Adam,
             learning_rate=1e-4,
             dropoutType = 'Dropout',
             act = 'softmax'):
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = Input(shape = (Samples, Chans, 1)) # treat multi-channel eeg as one frame of 2d-image

    ##################################################################
    block1       = Conv2D(F1, (kernLength,1), padding = 'same',
                                   use_bias = False)(input1)
    
#    block1_2       = Conv2D(F1, (kernLength//2,1), padding = 'same',
#                                   use_bias = False)(input1)
#    block1_3       = Conv2D(F1, (kernLength*2,1), padding = 'same',
#                                   use_bias = False)(input1)
#    block1_4       = Conv2D(F1, (kernLength*4,1), padding = 'same',
#                                   use_bias = False)(input1)
#    block1       = concatenate([block1, block1_2, block1_3, block1_4], axis=-1)
    
    block1       = BatchNormalization(axis = -1)(block1)
    block1       = DepthwiseConv2D((Chans,1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization(axis = -1)(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((4,1))(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, (16,1),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization(axis = -1)(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((8, 1))(block2)
    block2       = dropoutType(dropoutRate)(block2)
        
    flatten      = Flatten(name = 'flatten')(block2)
    
#    dense        = Dense(32, name = 'dense_0', 
#                         kernel_constraint = max_norm(0.5))(flatten)
#    dense        = Activation('elu')(dense)
    
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = Activation(act, name = 'softmax')(dense)
    
    Mymodel = Model(input1, softmax)
    Mymodel.compile(loss='categorical_crossentropy', 
                    metrics=['accuracy'],
                    optimizer=optimizer(lr=learning_rate))
    
    return Mymodel

#from keras.layers import GaussianNoise
def My_eeg_net_1d(nb_classes, Chans = 64, Samples = 128, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, 
             optimizer = Adam,
             learning_rate=1e-4,
             dropoutType = 'Dropout',
             act = 'softmax'):
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = Input(shape = (Samples, Chans), name = 'input') # treat multi-channel eeg as one frame of 2d-image

    ##################################################################
#    block1       = GaussianNoise(0.2)(input1)
    block1       = Conv1D(F1, kernLength, padding = 'same',
                                   use_bias = False, name = 'block1-C1D')(input1)  
#    block1 = My_LSTM(block1, 32, 1, 32, inter_dim_list=None)
    
    block1       = BatchNormalization(name = 'block1-BN1')(block1)
    block1       = SeparableConv1D(F2, 8, use_bias = False,  #no Depthwise1D
                                   depthwise_constraint = max_norm(1.),
                                   padding = 'same', name = 'block1-SC1D')(block1)
    block1       = BatchNormalization(name = 'block1-BN2')(block1)
    block1       = Activation('elu', name = 'block1-AC')(block1)
#    block1       = AveragePooling1D(8)(block1)
    block1       = MaxPooling1D(8, name = 'block1-MP')(block1)
    block1       = dropoutType(dropoutRate, name = 'block1-DP')(block1)
    
    block2       = SeparableConv1D(F2, 8,
                                   use_bias = False, padding = 'same', name = 'block2-SC1D')(block1)
    block2       = BatchNormalization(name = 'block2-BN')(block2)
    block2       = Activation('elu', name = 'block2-AC')(block2)
    block2       = AveragePooling1D(4, name = 'block2-AP')(block2)
#    block2       = MaxPooling1D(4)(block2)
    block2       = dropoutType(dropoutRate, name = 'block2-DP')(block2)
    
#    block2       = SeparableConv1D(F2, 5,
#                                   use_bias = False, padding = 'same')(block2)
#    block2       = BatchNormalization()(block2)
#    block2       = Activation('elu')(block2)
#    block2       = AveragePooling1D(2)(block2)
#    block2       = dropoutType(dropoutRate)(block2) 
#    
#    block2       = SeparableConv1D(F2, 2,
#                                   use_bias = False, padding = 'same')(block2)
#    block2       = BatchNormalization()(block2)
#    block2       = Activation('elu')(block2)
#    block2       = dropoutType(dropoutRate)(block2)

        
    flatten      = Flatten(name = 'flatten')(block2)
#    flatten      = GlobalAveragePooling1D()(block2)
    
#    dense        = Dense(nb_classes, 
#                         kernel_constraint = max_norm(norm_rate), activation = 'relu')(flatten)
    
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    
#    dense        = add([dense, dense1])
    softmax      = Activation(act, name = 'softmax')(dense)
    
    Mymodel = Model(input1, softmax)
    Mymodel.compile(loss='categorical_crossentropy', 
                    metrics=['accuracy'],
                    optimizer=optimizer(lr=learning_rate))
    
    return Mymodel


from modules import Resample, Resample_multi_channel
from loss import cross_reg
from keras.initializers import RandomNormal

def locnet(Samples, Chans, kernLength, output_channels = 1, pooling = None, activation='linear'):
    '''
    define the resampling net
    '''
    input1   = Input(shape = (Samples, Chans))
    
    block_re       = Conv1D(Chans, kernLength, padding = 'same',
                                   use_bias = False)(input1)   
    block_re        = BatchNormalization()(block_re)
    block_re       = Activation('elu')(block_re)
    block_re       = SeparableConv1D(output_channels, kernLength, use_bias = True, padding = 'same', 
#                                   depthwise_constraint = max_norm(1.), 
                                   strides=1,
#                                   kernel_initializer=RandomNormal(mean=0.0, stddev=1e-1),
                                   activation = activation)(block_re)
    
    if pooling is not None:
        block_re = AveragePooling1D(pooling)(block_re)
    
    return Model(input1, block_re, name='Resampler')

def loc_Unet(Samples, Chans, kernLength, output_channels = 1, pooling = None, activation = 'linear'):
    input1   = Input(shape = (Samples, Chans))
    
    block_re       = SeparableConv1D(Chans, kernLength, padding = 'same',
                                   use_bias = True)(input1)     
    block_re       = BatchNormalization()(block_re)
    block_re       = Activation('elu')(block_re)
    
    '''
    DownSampling path
    '''
    down_1         = MaxPooling1D(4)(block_re)
    down_1         = SeparableConv1D(Chans*2, kernLength, padding = 'same',
                                   use_bias = True)(down_1)     
    down_1         = BatchNormalization()(down_1)
    down_1         = Activation('elu')(down_1)
    
    down_2         = MaxPooling1D(2)(down_1)
    down_2         = SeparableConv1D(Chans*2, kernLength, padding = 'same',
                                   use_bias = True)(down_2)     
    down_2         = BatchNormalization()(down_2)
    down_2         = Activation('elu')(down_2)
    
    down_3         = MaxPooling1D(2)(down_2)
    down_3         = SeparableConv1D(Chans*4, kernLength, padding = 'same',
                                   use_bias = True)(down_3)     
    down_3         = BatchNormalization()(down_3)
    down_3         = Activation('elu')(down_3)
    
    '''
    Up sampling path
    '''
    up_2           = UpSampling1D(2)(down_3)
    up_2           = concatenate([up_2, down_2], axis=-1)
    up_2           = SeparableConv1D(Chans*2, kernLength, padding = 'same',
                                   use_bias = True)(up_2)     
    up_2           = BatchNormalization()(up_2)
    up_2           = Activation('elu')(up_2)   
    
    up_1           = UpSampling1D(2)(up_2)
    up_1           = concatenate([up_1, down_1], axis=-1)
    up_1           = SeparableConv1D(Chans*2, kernLength, padding = 'same',
                                   use_bias = True)(up_1)     
    up_1           = BatchNormalization()(up_1)
    up_1           = Activation('elu')(up_1)
    
    up             = UpSampling1D(4)(up_1)
    up             = concatenate([up, block_re], axis=-1)
    up             = SeparableConv1D(Chans, kernLength, padding = 'same',
                                   use_bias = True)(up)     
    up             = BatchNormalization()(up)
    up             = Activation('elu')(up)
    
    out            = SeparableConv1D(output_channels, 1, use_bias = True, padding = 'same', 
                                   depthwise_constraint = max_norm(1.),
                                   kernel_initializer=RandomNormal(mean=0.0, stddev=1e-1),
                                   activation = activation)(up)
    if pooling is not None:
        out = AveragePooling1D(pooling)(out)
    
    return Model(input1, out, name='Resampler')



def My_eeg_net_1d_resample(Sampler, Classifier, t_length, Chans, optimizer, loss_weights, share=False, pooling=1):
    
    
    _input   = Input(shape = (t_length, Chans))   
    
    
    '''
    resample
    '''
    if share:
        resampled_signal = Resample_multi_channel(localization_net = Sampler, 
                                output_size=(t_length//pooling, Chans))(_input)
    else:
        resampled_signal = Resample(localization_net = Sampler, 
                                output_size=(t_length//pooling, Chans))(_input) # need to configure output_size
    
    
    '''
    connect to the rest classification part
    '''    
    pred = Classifier(resampled_signal) # a second layer and then weighted sum?
    

    
    Mymodel = Model(_input, [pred, Sampler(_input)], name='Whole_Model')
    Mymodel.layers[-2].name = 'Classifier'
    Mymodel.layers[-1].name = 'Resampler'
    
    
    Mymodel.compile(loss=['categorical_crossentropy', cross_reg], 
                    loss_weights = loss_weights,
                    metrics=['accuracy'],
                    optimizer=optimizer)
    
    return Mymodel

from keras.layers import multiply
def My_eeg_net_pt_attd(Sampler, Classifier, t_length, Chans, optimizer, loss_weights):
    
    
    _input   = Input(shape = (t_length, Chans))   
    
    
    '''
    resample
    '''
    att = Sampler(_input)
    
    signal_attention = multiply([att, _input])
    
    
    '''
    connect to the rest classification part
    '''    
    pred = Classifier(signal_attention) # a second layer and then weighted sum?
    

    
    Mymodel = Model(_input, [pred, att], name='Whole_Model')
    Mymodel.layers[-1].name = 'Classifier'
    Mymodel.layers[1].name = 'Attention'
    
    
    Mymodel.compile(loss=['categorical_crossentropy', 'mse'], 
                    loss_weights = loss_weights,
                    metrics=['accuracy'],
                    optimizer=optimizer)
    
    return Mymodel

from modules import mask, band_mask
from keras.layers import ZeroPadding1D
def My_eeg_net_pt_attd_2(Sampler, Classifier, t_length, Chans, optimizer, loss_weights, thres = 0.5):
    
    
    _input   = Input(shape = (t_length, Chans))   
    
    
    '''
    resample
    '''
    att = Sampler(_input)
    
    signal_attention = mask(thres=0.5)([_input, att])

    
    '''
    connect to the rest classification part
    '''    
    pred = Classifier(signal_attention) # a second layer and then weighted sum?
    

    
    Mymodel = Model(_input, [pred, att], name='Whole_Model')
    Mymodel.layers[-1].name = 'Classifier'
    Mymodel.layers[1].name = 'Attention'
    
    
    Mymodel.compile(loss=['categorical_crossentropy', 'mse'], 
                    loss_weights = loss_weights,
                    metrics=['accuracy'],
                    optimizer=optimizer)
    
    return Mymodel

def My_eeg_net_freq_selection(Sampler, Classifier, t_length, Chans, optimizer, loss_weights, thres = 0.5, mask_type='hard'):
    
    
    _input   = Input(shape = (t_length, Chans))   
    
    
    '''
    resample
    '''
    att = Sampler(_input)
    
#    signal_attention = mask(thres=0.5)([_input, att])
    att = ZeroPadding1D((0,1))(att)
    signal_attention = band_mask(thres, mask_type)([_input, att])
#    print(signal_attention.shape)
    
    '''
    connect to the rest classification part
    '''    
    pred = Classifier(signal_attention) # a second layer and then weighted sum?
    

    
    Mymodel = Model(_input, [pred, att], name='Whole_Model')
    Mymodel.layers[-1].name = 'Classifier'
    Mymodel.layers[1].name = 'Attention'
    
    
    Mymodel.compile(loss=['categorical_crossentropy', 'mse'], 
                    loss_weights = loss_weights,
                    metrics=['accuracy'],
                    optimizer=optimizer)
    
    return Mymodel

from modules import Window_trunc, Window_trunc_no_weights, SpatialTransformer, STN_1D, STN_1D_noweights, STN_1D_noweights_multi_channel
def locnet_window(Samples, Chans, kernLength, share_w_channels=True):
    '''
    Take the input signal and outputs the starting points of truncated windows
    
    The output value is between 0 and 1
    '''
    input1         = Input(shape = (Samples, Chans))
    
    block_re       = SeparableConv1D(16, kernLength, padding = 'same',
                                   use_bias = True)(input1)     
    block_re       = BatchNormalization()(block_re)
    block_re       = Activation('elu')(block_re)
    
    '''
    DownSampling path
    '''
    down_1         = MaxPooling1D(4)(block_re)
    down_1         = SeparableConv1D(16, kernLength, padding = 'same',
                                   use_bias = True)(down_1)     
    down_1         = BatchNormalization()(down_1)
    down_1         = Activation('elu')(down_1)
    
#    down_2         = MaxPooling1D(2)(down_1)
#    down_2         = SeparableConv1D(32, kernLength, padding = 'same',
#                                   use_bias = True)(down_2)     
#    down_2         = BatchNormalization()(down_2)
#    down_2         = Activation('elu')(down_2)
#    
#    down_3         = MaxPooling1D(2)(down_2)
#    down_3         = SeparableConv1D(64, kernLength, padding = 'same',
#                                   use_bias = True)(down_3)     
#    down_3         = BatchNormalization()(down_3)
#    down_3         = Activation('elu')(down_3)
    
    flatten        = SeparableConv1D(1, 1, use_bias = True, padding = 'valid')(down_1)
    flatten        = Flatten()(flatten)
    
    if share_w_channels:                               
        out            = Dense(2, name = 'dense',
    #                           kernel_initializer=RandomNormal(mean=1.0, stddev=0.1),
                               activation = 'sigmoid'
                               )(flatten) # activation matters here
#    out            = Activation('relu', name = 'sigmoid')(out)
#        out_slope      = Dense(1, name = 'slope',
#    #                           kernel_initializer=RandomNormal(mean=1.0, stddev=0.1),
#                               activation = 'sigmoid'
#                               )(flatten)
#        out_shift      = Dense(1, name = 'shift',
#    #                           kernel_initializer=RandomNormal(mean=0.0, stddev=1.0),
#                               activation = 'tanh'
#                               )(flatten)
#        
#        out            = concatenate([out_slope, out_shift], axis=-1)

    else:
        out            = Dense(2*Chans, name = 'dense',
    #                           kernel_initializer=RandomNormal(mean=1.0, stddev=0.1),
                               activation = 'sigmoid'
                               )(flatten)
    
#        out_slope      = Dense(Chans, name = 'slope',
#    #                           kernel_initializer=RandomNormal(mean=1.0, stddev=0.1),
#                               activation = 'sigmoid'
#                               )(flatten)
#        out_shift      = Dense(Chans, name = 'shift',
#    #                           kernel_initializer=RandomNormal(mean=0.0, stddev=1.0),
#                               activation = 'tanh'
#                               )(flatten)
#        
#        out            = concatenate([out_slope, out_shift], axis=-1)
    
    
    return Model(input1, out, name='Window')


def My_eeg_net_window(Window, Classifier, t_length, window_len, Chans, optimizer, 
                      share_w_channels = True):
    
    
    _input   = Input(shape = (t_length, Chans))   
    
    
    '''
    Truncate the singal
    '''
#    windowed_signal = Window_trunc(localization_net = Window, 
#                                output_size=(window_len, Chans))(_input)
#    windowed_signal = SpatialTransformer(localization_net = Window, 
#                                output_size=(window_len, Chans))(_input)
#    windowed_signal = STN_1D(localization_net = Window, 
#                                output_size=(window_len, Chans))(_input)
    
    trans = Window(_input)
    if share_w_channels:
        windowed_signal = STN_1D_noweights(output_size=(window_len, Chans))([trans, _input])
    else:
        windowed_signal = STN_1D_noweights_multi_channel(output_size=(window_len, Chans))([trans, _input])
      
#    proposed_starts  = Window(_input)
#    windowed_signal = Window_trunc_no_weights(output_size=(window_len, Chans))([_input, proposed_starts])
    
    
    '''
    connect to the rest classification part
    '''    
    pred = Classifier(windowed_signal) # a second layer and then weighted sum?
    

    
    Mymodel = Model(_input, pred, name='Whole_Model')
    Mymodel.layers[-1].name = 'Classifier'
    Mymodel.layers[-2].name = 'Window'
    
    
    Mymodel.compile(loss='categorical_crossentropy', 
#                    loss_weights = loss_weights,
                    metrics=['accuracy'],
                    optimizer=optimizer)
    
    return Mymodel

    

def EEGNet(nb_classes, Chans = 64, Samples = 128, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout', act = 'softmax'):
    """ Keras Implementation of EEGNet
    http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta
    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version. For example:
        
        1. Depthwise Convolutions to learn spatial filters within a 
        temporal convolution. The use of the depth_multiplier option maps 
        exactly to the number of spatial filters learned within a temporal
        filter. This matches the setup of algorithms like FBCSP which learn 
        spatial filters within each filter in a filter-bank. This also limits 
        the number of free parameters to fit when compared to a fully-connected
        convolution. 
        
        2. Separable Convolutions to learn how to optimally combine spatial
        filters across temporal bands. Separable Convolutions are Depthwise
        Convolutions followed by (1x1) Pointwise Convolutions. 
        
    
    While the original paper used Dropout, we found that SpatialDropout2D 
    sometimes produced slightly better results for classification of ERP 
    signals. However, SpatialDropout2D significantly reduced performance 
    on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
    the default Dropout in most cases.
        
    Assumes the input signal is sampled at 128Hz. If you want to use this model
    for any other sampling rate you will need to modify the lengths of temporal
    kernels and average pooling size in blocks 1 and 2 as needed (double the 
    kernel lengths for double the sampling rate, etc). Note that we haven't 
    tested the model performance with this rule so this may not work well. 
    
    The model with default parameters gives the EEGNet-8,2 model as discussed
    in the paper. This model should do pretty well in general, although it is
	advised to do some model searching to get optimal performance on your
	particular dataset.
    We set F2 = F1 * D (number of input filters = number of output filters) for
    the SeparableConv2D layer. We haven't extensively tested other values of this
    parameter (say, F2 < F1 * D for compressed learning, and F2 > F1 * D for
    overcomplete). We believe the main parameters to focus on are F1 and D. 
    Inputs:
        
      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.     
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D. 
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.
    """
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = Input(shape = (1, Chans, Samples))

    ##################################################################
    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   use_bias = False)(input1)
    block1       = BatchNormalization(axis = 1)(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization(axis = 1)(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization(axis = 1)(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)
        
    flatten      = Flatten(name = 'flatten')(block2)
    
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = Activation(act, name = 'softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)

