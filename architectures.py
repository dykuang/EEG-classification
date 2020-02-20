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
from keras import regularizers

# model

def TC_arc_sep(time_steps, num_channels, num_classes, num_down=None, optimizer=Adam, learning_rate=1e-3):
    '''
    Temporal convolutional blocks + MLP block
    '''
    
    x_in = Input(shape=(time_steps, num_channels))
    
    
    x1 = Conv1D(16, kernel_size=30, strides = 1, padding='same',use_bias = False)(x_in)
    x2 = Conv1D(16, kernel_size=13, strides = 1, padding='same',use_bias = False)(x_in)
    x3 = Conv1D(16, kernel_size=3, strides = 1, padding='same',use_bias = False)(x_in)
    x4 = Conv1D(16, kernel_size=7, strides = 1, padding='same',use_bias = False)(x_in)
    x = concatenate([x1,x2,x3,x4], axis=-1)
    x = LeakyReLU(name='LR_1')(x)   
        
    x = SeparableConv1D(32, kernel_size=3, strides = 4,depth_multiplier=1,use_bias = False)(x)
    x = LeakyReLU(name='LR_2')(x)
    x = SpatialDropout1D(0.25)(x)
    
    x = SeparableConv1D(32, kernel_size=3, strides = 1,depth_multiplier=1,use_bias = False)(x)
    x = LeakyReLU(name='LR_3')(x)
    x = SeparableConv1D(32, kernel_size=3, strides = 4,depth_multiplier=1,use_bias = False)(x)
    x = LeakyReLU(name='LR_4')(x)
    x = SpatialDropout1D(0.25)(x)
#    x = My_LSTM(x, 32, 1, 32, inter_dim_list=None)
    
    if num_down is not None:
        for i in range(num_down):
                x = SeparableConv1D(32, kernel_size=3, strides = 1,depth_multiplier=1,use_bias = False)(x)
                x = LeakyReLU()(x)
                x = SeparableConv1D(32, kernel_size=3, strides = 2,depth_multiplier=1,use_bias = False)(x)
                x = LeakyReLU()(x)
    
    
    x = Flatten()(x)
    x = Dense(64, activation = 'relu', name = 'dense_1', kernel_constraint = max_norm(1.0))(x)
    #x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Dense(16, activation = 'relu', name = 'dense',kernel_constraint = max_norm(0.5))(x)
    #x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x_out = Dense(num_classes, activation = 'softmax', name = 'output')(x)
    
    Mymodel = Model(x_in, x_out)
    
    Mymodel.compile(loss='categorical_crossentropy', 
                    metrics=['accuracy'],
                    optimizer=optimizer(lr=learning_rate))
    
    return Mymodel

def TC_arc_1(time_steps, num_channels, num_classes, 
             num_down=None, optimizer=Adam, learning_rate=1e-3):
    '''
    A depthwise conv version of arc_1
    '''
    
    x_in = Input(shape=(time_steps, num_channels))
    
    
    x1 = Conv1D(16, kernel_size=30, strides = 1, padding='same',use_bias = False)(x_in)
    x2 = Conv1D(16, kernel_size=13, strides = 1, padding='same',use_bias = False)(x_in)
    x3 = Conv1D(16, kernel_size=3, strides = 1, padding='same',use_bias = False)(x_in)
    x4 = Conv1D(16, kernel_size=7, strides = 1, padding='same',use_bias = False)(x_in)
    x = concatenate([x1,x2,x3,x4], axis=-1)
    x = LeakyReLU(name='LR_1')(x)   
        
    x = Conv1D(32, kernel_size=3, strides = 2,use_bias = False)(x)
    x = LeakyReLU(name='LR_2')(x)
      
    x = Conv1D(32, kernel_size=3, strides = 1,use_bias = False)(x)
    x = LeakyReLU(name='LR_3')(x)
    x = Conv1D(32, kernel_size=3, strides = 2,use_bias = False)(x)
    x = LeakyReLU(name='LR_4')(x)
    
#    x = My_LSTM(x, 32, 1, 32, inter_dim_list=None)
    
    if num_down is not None:
        for i in range(num_down):
                x = Conv1D(32, kernel_size=3, strides = 1)(x)
                x = LeakyReLU()(x)
                x = Conv1D(32, kernel_size=3, strides = 2)(x)
                x = LeakyReLU()(x)
    
    
    x = Flatten()(x)
    x = Dense(64, activation = 'relu', name = 'dense_1', kernel_constraint = max_norm(1.0))(x)
    #x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(16, activation = 'relu', name = 'dense', kernel_constraint = max_norm(0.5))(x)
    #x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x_out = Dense(num_classes, activation = 'softmax', name = 'output')(x)
    
    Mymodel = Model(x_in, x_out)
    
    Mymodel.compile(loss='categorical_crossentropy', 
                    metrics=['accuracy'],
                    optimizer=optimizer(lr=learning_rate))
    
    return Mymodel

def TC_arc_2(time_steps, num_channels, num_classes, optimizer=Adam, learning_rate=1e-3):
    '''
    A variation from above with one extra input block
    '''
    x_in = Input(shape=(time_steps, num_channels))
       
    x1 = Conv1D(16, kernel_size=3, strides = 1, padding='same')(x_in)
    x2 = Conv1D(16, kernel_size=1, strides = 1, padding='same')(x_in)
    x3 = Conv1D(16, kernel_size=5, strides = 1, padding='same')(x_in)
    x4 = Conv1D(16, kernel_size=7, strides = 1, padding='same')(x_in)
    x = concatenate([x1,x2,x3,x4], axis=-1)
    x = LeakyReLU(name='LR_1')(x)
    
    x_in_2 = Input(shape=(time_steps, num_channels))
    x1_2 = Conv1D(16, kernel_size=3, strides = 1, padding='same')(x_in_2)
    x2_2 = Conv1D(16, kernel_size=1, strides = 1, padding='same')(x_in_2)
    x3_2 = Conv1D(16, kernel_size=5, strides = 1, padding='same')(x_in_2)
    x4_2 = Conv1D(16, kernel_size=7, strides = 1, padding='same')(x_in_2)
    x_2 = concatenate([x1_2,x2_2,x3_2,x4_2], axis=-1)
    x_2 = LeakyReLU()(x_2)
    
    x = concatenate([x, x_2], axis=-1)
    
        
    x = Conv1D(32, kernel_size=3, strides = 2)(x)
    x = LeakyReLU(name='LR_2')(x)
    
    
    x = Conv1D(32, kernel_size=3, strides = 1)(x)
    x = LeakyReLU(name='LR_3')(x)
    x = Conv1D(32, kernel_size=3, strides = 2)(x)
    x = LeakyReLU(name='LR_4')(x)
    
    
    x = Flatten()(x)
    x = Dense(64, activation = 'relu', name = 'relu_1')(x)
    #x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(16, activation = 'relu', name = 'relu_2')(x)
    #x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x_out = Dense(num_classes, activation = 'softmax')(x)
    
    Mymodel = Model(x_in, x_out)
    
    Mymodel.compile(loss='categorical_crossentropy', 
                    metrics=['accuracy'],
                    optimizer=optimizer(lr=learning_rate))
    
    return Mymodel

def TC_arc_3(time_steps, num_channels, num_classes, optimizer=Adam, learning_rate=1e-3):
    '''
    Try the leaf classification net
    '''
    
    x_in = Input(shape=(time_steps, num_channels))
    x = Conv1D(filters= 16, kernel_size = 8, strides=4, padding='same', dilation_rate=1, 
           activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
           bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
           activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
           name = 'conv1D_1')(x_in)
    x = BatchNormalization()(x)
    #x = PReLU()(x)
    x = MaxPooling1D(pool_size=4, strides=2, name = 'MP_1')(x)
    x = Flatten(name = 'flat_1')(x)
    
    x_x = Conv1D(filters= 24, kernel_size = 12, strides= 6, padding='same', dilation_rate=1, 
           activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
           bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
           activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
           name = 'conv1D_2')(x_in)
    x_x = BatchNormalization()(x_x)
    #x_x = PReLU()(x_x)
    x_x = MaxPooling1D(pool_size=4, strides=2, name = 'MP_2')(x_x)
    x_x = Flatten(name = 'flat_2')(x_x)
    
    x_x_x = Conv1D(filters= 32, kernel_size = 16, strides= 8, padding='same', dilation_rate=1, 
           activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
           bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
           activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
           name = 'conv1D_3')(x_in)
    x_x_x = BatchNormalization()(x_x_x)

    x_x_x = MaxPooling1D(pool_size=4, strides=2, name = 'MP_3')(x_x_x)
    x_x_x = Flatten(name = 'flat_3')(x_x_x)

    feature_f = Flatten(name = 'flat_4')(x_in)
    #
    x = concatenate([x, x_x, x_x_x, feature_f])
    #x = BatchNormalization()(x) 
    #x = Dropout(0.5)(x)
    
    x = Dense(512, activation = 'linear', name = 'dense_1')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    #x = Dropout(0.5)(x)
    
    x = Dense(128, activation = 'linear', name = 'dense_2')(x) #increase the dimension here for better speration in stage2 ?
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Dropout(0.5)(x)
    
    
    pred = Dense(num_classes, activation = 'softmax', name = 'dense_3')(x)
    
    Mymodel = Model(x_in, pred)
    
    Mymodel.compile(loss='categorical_crossentropy', 
                    metrics=['accuracy'],
                    optimizer=optimizer(lr=learning_rate))
    
    return Mymodel


def TC_arc_4(time_steps, num_channels, num_classes, optimizer=Adam, learning_rate=1e-3):
    '''
    Temporal convolutional blocks (deep) + MLP block
    '''
    
    x_in = Input(shape=(time_steps, num_channels))
    
    
    x1 = Conv1D(16, kernel_size=30, strides = 1, padding='same')(x_in)
    x2 = Conv1D(16, kernel_size=11, strides = 1, padding='same')(x_in)
    x3 = Conv1D(16, kernel_size=5, strides = 1, padding='same')(x_in)
    x4 = Conv1D(16, kernel_size=7, strides = 1, padding='same')(x_in)
    x = concatenate([x1,x2,x3,x4], axis=-1)
    x = LeakyReLU(name='LR_1')(x)   
        
    x = Conv1D(32, kernel_size=3, strides = 2)(x)
    x = LeakyReLU(name='LR_2')(x)
      
    x = Conv1D(32, kernel_size=3, strides = 1)(x)
    x = LeakyReLU(name='LR_3')(x)
    x = Conv1D(32, kernel_size=3, strides = 2)(x)
    x = LeakyReLU(name='LR_4')(x)
    
    for _ in range(7):
            x_b = Conv1D(32, kernel_size=3, strides = 1, padding='same')(x)
            x_b = LeakyReLU()(x_b)
            x = add([x_b, x])
            x = BatchNormalization()(x)
        
    x = Flatten()(x)
    x = Dense(64, activation = 'relu', name = 'dense_1')(x)
    #x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(16, activation = 'relu', name = 'dense_2')(x)
    #x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x_out = Dense(num_classes, activation = 'softmax', name = 'output')(x)
    
    Mymodel = Model(x_in, x_out)
    
    Mymodel.compile(loss='categorical_crossentropy', 
                    metrics=['accuracy'],
                    optimizer=optimizer(lr=learning_rate, decay=1e-3))
    
    return Mymodel

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
    
    input1   = Input(shape = (Samples, Chans)) # treat multi-channel eeg as one frame of 2d-image

    ##################################################################
#    block1       = GaussianNoise(0.2)(input1)
    block1       = Conv1D(F1, kernLength, padding = 'same',
                                   use_bias = False)(input1)  
    
#    block1 = My_LSTM(block1, 32, 1, 32, inter_dim_list=None)
    
    block1       = BatchNormalization()(block1)
#    block1       = Activation('elu')(block1)

    block1       = SeparableConv1D(F2, 8, use_bias = False,  #no Depthwise1D
                                   depthwise_constraint = max_norm(1.),
                                   padding = 'same')(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling1D(8)(block1)
#    block1       = MaxPooling1D(8)(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv1D(F2, 8, use_bias = False,
#                                   depthwise_constraint = max_norm(1.),
                                   padding = 'same')(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling1D(4)(block2)
#    block2       = MaxPooling1D(4)(block2)
    block2       = dropoutType(dropoutRate)(block2)
    
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
    
#    dense        = Dense(32, kernel_constraint = max_norm(norm_rate), name='feature', activation = 'elu')(flatten)
    
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate)
                         )(flatten)
    
#    dense        = add([dense, dense1])
    softmax      = Activation(act, name = 'softmax')(dense)
    
    Mymodel = Model(input1, softmax)
    Mymodel.compile(loss='categorical_crossentropy', 
                    metrics=['accuracy'],
                    optimizer=optimizer(lr=learning_rate))
    
    return Mymodel


def My_eeg_net_1d_w_att(nb_classes, Chans = 64, Samples = 128, 
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
    
    input1   = Input(shape = (Samples, Chans)) # treat multi-channel eeg as one frame of 2d-image

    ##################################################################
#    block1       = GaussianNoise(0.2)(input1)
    block1       = Conv1D(F1, kernLength, padding = 'same',
                                   use_bias = False)(input1)  
    
    block1       = BatchNormalization()(block1)
    block1       = SeparableConv1D(F2, 8, use_bias = False,  #no Depthwise1D
                                   depthwise_constraint = max_norm(1.),
                                   padding = 'same')(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
#    block1       = AveragePooling1D(8)(block1)
    block1       = MaxPooling1D(8)(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
#====================================================
# The attantion block
#==================================================    
#    att          = locnet(12, 64, 3, 
#                          output_channels = 64, pooling = None, 
#                          activation='sigmoid')(block2)
    block_re       = Conv1D(32, 3, padding = 'same',
                                   use_bias = True)(block1)   
    block_re       = BatchNormalization()(block_re)
    block_re       = Activation('elu')(block_re)
    att            = SeparableConv1D(64, 3, use_bias = True, padding = 'same', 
#                                   depthwise_constraint = max_norm(1.), 
                                   strides=1,
#                                   kernel_initializer=RandomNormal(mean=0.0, stddev=1e-1),
                                   activity_regularizer=regularizers.l1(1e-5),
                                   activation = 'sigmoid')(block_re)
    
#    block2       = mask(0.5)([block2, att])  # reports error ?
    block1       = multiply([block1, att])
#========================================================
    
    block2       = SeparableConv1D(F2, 8,
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling1D(4)(block2)
#    block2       = MaxPooling1D(4)(block2)
    block2       = dropoutType(dropoutRate)(block2)
  
##====================================================
## The attantion block
##==================================================    
##    att          = locnet(12, 64, 3, 
##                          output_channels = 64, pooling = None, 
##                          activation='sigmoid')(block2)
#    block_re2       = Conv1D(64, 3, padding = 'same',
#                                   use_bias = True)(block2)   
#    block_re2       = BatchNormalization()(block_re2)
#    block_re2       = Activation('elu')(block_re2)
#    att2            = SeparableConv1D(64, 3, use_bias = True, padding = 'same', 
##                                   depthwise_constraint = max_norm(1.), 
#                                   strides=1,
##                                   kernel_initializer=RandomNormal(mean=0.0, stddev=1e-1),
#                                   activity_regularizer=regularizers.l1(1e-4),
#                                   activation = 'sigmoid')(block_re2)
#    
##    block2       = mask(0.5)([block2, att])  # reports error ?
#    block2       = multiply([block2, att2])
##========================================================
       
    flatten      = Flatten(name = 'flatten')(block2)

    
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    
#    dense        = add([dense, dense1])
    softmax      = Activation(act, name = 'softmax')(dense)
    
    Mymodel = Model(input1, softmax)
    Mymodel.compile(loss='categorical_crossentropy', 
                    metrics=['accuracy'],
                    optimizer=optimizer(lr=learning_rate))
    
    return Mymodel


def My_eeg_net_1d_w_CM(nb_classes, Chans = 64, Samples = 128, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, 
             optimizer = Adam,
             learning_rate=1e-4,
             dropoutType = 'Dropout',
             act = 'softmax',
             CM_shape = (19, 45) ):
    '''
    The 1d eeg net with static connectitvity matrix as extra input
    '''
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = Input(shape = (Samples, Chans)) 
#    input2   = Input(shape = (Chans, Chans, 1))
#    input2   = Input(shape = (45,))
    input2   = Input(shape = CM_shape)
    
    ##################################################################
    # The 1d branch
    ###################################################################
#    block1       = GaussianNoise(0.2)(input1)
    block1       = Conv1D(F1, kernLength, padding = 'same',
                                   use_bias = False)(input1)  
#    block1 = My_LSTM(block1, 32, 1, 32, inter_dim_list=None)
    
    block1       = BatchNormalization()(block1)
#    block1       = Activation('elu')(block1)
    
    block1       = SeparableConv1D(F2, 8, use_bias = False,  #no Depthwise1D
                                   depthwise_constraint = max_norm(1.),
                                   padding = 'same')(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling1D(8)(block1)
#    block1       = MaxPooling1D(8)(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv1D(F2, 8,
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling1D(4)(block2)
#    block2       = MaxPooling1D(4)(block2)
    block2       = dropoutType(dropoutRate)(block2)

        
    flatten1      = Flatten(name = 'flatten1')(block2)
    
    #####################################################################
    # The 2d branch
    #####################################################################
# =============================================================================
#     block       = Conv2D(32, 3, padding= 'same', use_bias = False)(input2)
#     block       = BatchNormalization(axis = -1)(block)
# #    block       = Activation('elu')(block)
#     block       = DepthwiseConv2D(2, use_bias = False, 
#                                    depth_multiplier = 1,
#                                    depthwise_constraint = None)(block)
#     block       = Activation('elu')(block)
#     block       = AveragePooling2D((2, 2))(block)
#     
#     block       = SeparableConv2D(64, 3, use_bias = False, 
#                                    depth_multiplier = 2,
#                                    depthwise_constraint = max_norm(1.))(block)
#     block       = Activation('elu')(block)
# #    block       = AveragePooling2D((2, 2))(block)
#     block       = dropoutType(dropoutRate)(block)
#         
#     flatten2    = Flatten(name = 'flatten2')(block)
# =============================================================================
#    dense2      = Dense(16, activation='elu')(flatten2)     
    
#    '''
#    only the lower left info
#    '''
#    flatten2    = Dense(128, activation = 'elu')(input2)
    
    '''
    if dcm
    '''
    block       = SeparableConv1D(64, 3, use_bias = False, 
                                   depth_multiplier = 1,
                                   depthwise_constraint = None)(input2)
    block       = Activation('elu')(block)
    block       = AveragePooling1D(3)(block)
    
    block       = SeparableConv1D(64, 3, use_bias = False, 
                                   depth_multiplier = 2,
                                   depthwise_constraint = max_norm(1.))(block)
    block       = Activation('elu')(block)
#    block       = AveragePooling2D((2, 2))(block)
    block       = dropoutType(dropoutRate)(block)
        
    flatten2    = Flatten(name = 'flatten2')(block)

    ######################################################################
    # Merge to classify
    ######################################################################
    
    flatten      = concatenate([flatten1, flatten2])
    
#    dense        = Dense(32, name = 'feature', kernel_constraint = max_norm(norm_rate), activation='elu')(flatten)
    
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    
    softmax      = Activation(act, name = 'softmax')(dense)
    
    Mymodel = Model([input1,input2], softmax)
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
                                   use_bias = True)(input1)   
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
    '''
    Point attention will be applied in a 'soft' way
    '''
    
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
    '''
    Point attention will be applied in a 'hard' way
    '''    
    
    _input   = Input(shape = (t_length, Chans))   
    
    
    '''
    resample
    '''
    att = Sampler(_input)
    
    signal_attention = mask(thres)([_input, att])

    
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

from modules import Window_trunc, Window_trunc_no_weights
def locnet_window(Samples, Chans, kernLength, norm_rate= 0.25):
    '''
    Take the input signal and outputs the starting points of truncated windows
    
    The output value is between 0 and 1
    '''
    input1         = Input(shape = (Samples, Chans))
    
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
    
    flatten        = SeparableConv1D(1, 1, use_bias = True, padding = 'valid')(down_3)
    flatten        = Flatten()(flatten)
                                   
    out            = Dense(Chans, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    out            = Activation('sigmoid', name = 'sigmoid')(out)
    
    return Model(input1, out, name='Window')


def My_eeg_net_window(Window, Classifier, t_length, window_len, Chans, optimizer):
    
    
    _input   = Input(shape = (t_length, Chans))   
    
    
    '''
    Truncate the singal
    '''
    windowed_signal = Window_trunc(localization_net = Window, 
                                output_size=(window_len, Chans))(_input)
    
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

import keras.backend as K
from keras.layers import Lambda, Reshape
from keras.losses import mean_squared_error as mse
from loss import myLoss, KL_loss
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


from modules import VAE_SamplingLayer
def My_eeg_net_vae(nb_classes, Chans = 64, Samples = 128, latent_dim = 1,
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
    
    input1   = Input(shape = (Samples, Chans)) 
    ##################################################################
    # encoder
    #=================================================================
#    block1       = GaussianNoise(0.2)(input1)
    block1       = SeparableConv1D(F1, kernLength, padding = 'same',
                                   use_bias = False)(input1)     
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
    
    block1       = SeparableConv1D(F2, 8, use_bias = False,  #no Depthwise1D
                                   depthwise_constraint = max_norm(1.),
                                   padding = 'same')(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling1D(8)(block1)
#    block1       = MaxPooling1D(8)(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv1D(F2, 8,
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling1D(4)(block2)
#    block2       = MaxPooling1D(4)(block2)
    block2       = dropoutType(dropoutRate)(block2)
            
    flatten      = Flatten(name = 'flatten')(block2)
    
#    flatten      = Dense(64)(flatten)
    
    z_mu     = Dense(latent_dim, name = 'z_mu', 
                         kernel_constraint = max_norm(norm_rate)
                         )(flatten)
    
    z_log_sigma  = Dense(latent_dim, name = 'z_sigma', 
                         kernel_constraint = max_norm(norm_rate)
                         )(flatten)
    
#    z = Lambda(sampling, output_shape=(nb_classes,), name='z')([z_mu, z_log_sigma])
    z = VAE_SamplingLayer()([z_mu, z_log_sigma])
    
    encoder = Model(input1, [z_mu, z_log_sigma, z], name='encoder')
    #======================================================================
    # decoder
    #=======================================================================
    
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    
    dense        = Dense(768)(latent_inputs)
#    dense         = Dense(768)(dense)
    
    reshaped     = Reshape((-1, F2))(dense)
    
    block2_T     = UpSampling1D(4)(reshaped)
    block2_T     = SeparableConv1D(F2, 8,
                                   use_bias = False, padding = 'same')(block2_T)
    block2_T     = BatchNormalization()(block2_T)
    block2_T     = Activation('elu')(block2_T)
    block2_T     = ZeroPadding1D(1)(block2_T)
    
    block1_T     = UpSampling1D(8)(block2_T)
    block1_T     = SeparableConv1D(F2, 8,
                                   use_bias = False, padding = 'same')(block1_T)
    block1_T     = BatchNormalization()(block1_T)
    block1_T     = Activation('elu')(block1_T) 
    
    
    decoder_output= SeparableConv1D(Chans, 1, activation='sigmoid', padding='same')(block1_T)

    decoder = Model(latent_inputs, decoder_output, name = 'decoder')
    
    
    
#    VAE     = Model(input1, [z_mu, decoder(encoder(input1)[2])], name = 'vae')
    
#    reconstruction_loss = mse(input1, decoder(encoder(input1)[2]))
#
#    kl_loss = 1 + z_log_sigma - K.square(z_mu) - K.exp(z_log_sigma)
#    kl_loss = K.sum(kl_loss, axis=-1)
#    kl_loss *= -0.5
#    vae_loss = K.mean(reconstruction_loss + kl_loss)
#    VAE.add_loss(vae_loss)

#    VAE.compile(loss=myLoss(), 
##                    loss_weights = [1.0, 1.0, 1.0],
#                    metrics=['accuracy'],
#                    optimizer=optimizer(lr=learning_rate))
    
#    VAE     = Model(input1, [z_mu, decoder(encoder(input1)[2])], name = 'vae')
#    VAE.compile(loss=['categorical_crossentropy', 'mse'], 
#                    loss_weights = [1.0, 1.0],
#                    metrics=['accuracy'],
#                    optimizer=optimizer(lr=learning_rate))
    
    VAE     = Model(input1, decoder(encoder(input1)[-1]), name = 'vae')
    VAE.compile(loss='mse', 
#                    metrics=['accuracy'],
                    optimizer=optimizer(lr=learning_rate))
    
    return VAE

