#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 14:18:09 2019

@author: dykuang

Some extra modules to be imported in architectures
"""

from keras.layers.core import Layer
import tensorflow as tf
#import tensorflow_probability as tfp

class Resample(Layer):
    '''
    Different channels will have different displacement
    '''
    def __init__(self,
                 localization_net, # this suppose to produce a deformation with 3 channels
                 output_size,
                 **kwargs):
        self.locnet = localization_net
        self.output_size = output_size
        super(Resample, self).__init__(**kwargs)

    def build(self, input_shape):
        self.locnet.build(input_shape)
        self.trainable_weights = self.locnet.trainable_weights
        super(Resample, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        output_size = self.output_size
        return (None,
                int(output_size[0]), # time
                int(output_size[1]), # channels
                )  

    def call(self, X, mask=None): 
        deformation = self.locnet.call(X)
#        Y = tf.expand_dims(X[...,0], 4) # only transform the first channel
        output = self._transform(deformation, X, self.output_size) 
        return output

    def _repeat(self, x, num_repeats): # copy along the second dimension, each row is a copy of an index
        ones = tf.ones((1, num_repeats), dtype='int32')
        x = tf.reshape(x, shape=(-1,1))
        x = tf.matmul(x, ones)
        return tf.reshape(x, [-1])

    def _interpolate(self, signal, x, output_size): 
        '''
        signal: (B, T, C)
        x: (B*C*T, )
        '''
        batch_size = tf.shape(signal)[0]
        t_len = tf.shape(signal)[1]
        num_channels = tf.shape(signal)[-1]

        x = tf.cast(x , dtype='float32')

        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        
        max_x = tf.cast(t_len - 1,  dtype='int32')        
        zero = tf.zeros([], dtype='int32')

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)

        pts_batch = tf.range(batch_size*num_channels)*t_len
#        pts_batch = tf.range(batch_size*t_len)*num_channels
#        pts_batch = tf.range(t_len)*batch_size*num_channels
        flat_output_dimensions = output_size[0]
        base = self._repeat(pts_batch, flat_output_dimensions)
        
        ind_0 = base + x0
        ind_1 = base + x1

        flat_signal = tf.transpose(signal, (0,2,1))
        flat_signal = tf.reshape(flat_signal, [-1] )
        flat_signal = tf.cast(flat_signal, dtype='float32')
        
       
        pts_values_0 = tf.gather(flat_signal, ind_0)
        pts_values_1 = tf.gather(flat_signal, ind_1)
        
        
        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')

        
        w_0 = x1 - x
        w_1 = x - x0

        output = w_0*pts_values_0 + w_1*pts_values_1
        
        output = tf.reshape(output, (-1, output_size[1], output_size[0]))
#        
        output = tf.transpose(output, (0, 2, 1) )
#        
     
        return output

    def _meshgrid(self, t_length):

        len_float = tf.cast(t_length, dtype='float32')
        indices_grid = tf.linspace(0.0, len_float - 1.0, self.output_size[0])
                                 
        return indices_grid

    def _transform(self, deformation, input_sig, output_size):
        batch_size = tf.shape(input_sig)[0]
        t_len = tf.shape(input_sig)[1]
        num_channels = tf.shape(input_sig)[-1]
              
        indices_grid = self._meshgrid(t_len)

        indices_grid = tf.tile(indices_grid, tf.stack([batch_size*num_channels]))
        indices_grid = tf.reshape(indices_grid, (batch_size, num_channels, -1) )
#        indices_grid = tf.transpose(indices_grid, (0, 2, 1))

#        deformation = tf.reshape(deformation, (-1, output_size, 3))
        deformation = tf.transpose(deformation, (0, 2, 1)) #(B, C, T)

#        transformed_grid = indices_grid  # for testing
        transformed_grid = indices_grid + deformation 
        
        x_s_flatten = tf.reshape(transformed_grid, [-1])

        transformed_vol = self._interpolate(input_sig, 
                                                x_s_flatten,
                                                output_size)

        
        return transformed_vol 
    
class Resample_multi_channel(Layer):
    '''
    Signals from different channels will share the same displacement
    Q1: windows of different sizes? Multiple windows
        * Only learn a fixed length
    Q2: How to make the correct slice
        * tf.gather
        * tf.linspace
        * tf.clip_by_value
        * tf.cast
    '''

    def __init__(self,
                 localization_net, # this suppose to produce a deformation with 3 channels
                 output_size,
                 **kwargs):
        self.locnet = localization_net
        self.output_size = output_size
        super(Resample_multi_channel, self).__init__(**kwargs)

    def build(self, input_shape):
        self.locnet.build(input_shape)
        self.trainable_weights = self.locnet.trainable_weights
        super(Resample_multi_channel, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        output_size = self.output_size
        return (None,
                int(output_size[0]), # time
                int(output_size[1]), # channels
                )  

    def call(self, X, mask=None): 
        deformation = self.locnet.call(X)
#        Y = tf.expand_dims(X[...,0], 4) # only transform the first channel
        output = self._transform(deformation, X, self.output_size) 
        return output

    def _repeat(self, x, num_repeats): # copy along the second dimension, each row is a copy of an index
        ones = tf.ones((1, num_repeats), dtype='int32')
        x = tf.reshape(x, shape=(-1,1))
        x = tf.matmul(x, ones)
        return tf.reshape(x, [-1])

    def _interpolate(self, signal, x, output_size): 
        '''
        signal: (B, T, C)
        x: (B*T, )
        '''
        batch_size = tf.shape(signal)[0]
        t_len = tf.shape(signal)[1]
        num_channels = tf.shape(signal)[-1]

        x = tf.cast(x , dtype='float32')

        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        
        max_x = tf.cast(t_len - 1,  dtype='int32')        
        zero = tf.zeros([], dtype='int32')

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)

        pts_batch = tf.range(batch_size)*t_len
        flat_output_dimensions = output_size[0]
        base = self._repeat(pts_batch, flat_output_dimensions)
        
        ind_0 = base + x0
        ind_1 = base + x1

#        flat_signal = tf.transpose(signal, (0,2,1))
        flat_signal = tf.reshape(signal, [-1, num_channels] )
        flat_signal = tf.cast(flat_signal, dtype='float32')
        
       
        pts_values_0 = tf.gather(flat_signal, ind_0)
        pts_values_1 = tf.gather(flat_signal, ind_1)
        
        
        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')

        
        w_0 = tf.expand_dims(x1 - x, 1)
        w_1 = tf.expand_dims(x - x0, 1)

        output = w_0*pts_values_0 + w_1*pts_values_1
        
        output = tf.reshape(output, (-1, output_size[0], output_size[1]))
          
     
        return output

    def _meshgrid(self, t_length):

        len_float = tf.cast(t_length, dtype='float32')
        indices_grid = tf.linspace(0.0, len_float - 1.0, self.output_size[0])
                                 
        return indices_grid

    def _transform(self, deformation, input_sig, output_size):
        batch_size = tf.shape(input_sig)[0]
        t_len = tf.shape(input_sig)[1]
#        num_channels = tf.shape(input_sig)[-1]
              
        indices_grid = self._meshgrid(t_len)

        indices_grid = tf.tile(indices_grid, tf.stack([batch_size]))
        indices_grid = tf.reshape(indices_grid, (batch_size, -1) )
#        indices_grid = tf.transpose(indices_grid, (0, 2, 1))

#        deformation = tf.reshape(deformation, (-1, output_size, 3))
#        deformation = tf.transpose(deformation, (0, 2, 1)) #(B, C, T)

#        transformed_grid = indices_grid  # for testing
        
        transformed_grid = tf.expand_dims(indices_grid, 2) + deformation 
        
        x_s_flatten = tf.reshape(transformed_grid, [-1])

        transformed_vol = self._interpolate(input_sig, 
                                                x_s_flatten,
                                                output_size)

        
        return transformed_vol     

class STN_1D_noweights_multi_channel(Layer):
    '''
    1D spatial transformer,
    Each channel has its own transformation
    '''

    def __init__(self,
#                 localization_net, # this suppose to produce a deformation with 3 channels
                 output_size,
                 **kwargs):
#        self.locnet = localization_net
        self.output_size = output_size
        super(STN_1D_noweights_multi_channel, self).__init__(**kwargs)

    def build(self, input_shape):
#        self.locnet.build(input_shape)
#        self.trainable_weights = self.locnet.trainable_weights
        super(STN_1D_noweights_multi_channel, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        output_size = self.output_size
        return (None,
                int(output_size[0]), # time
                int(output_size[1]), # channels
                )  

    def call(self, X, mask=None): 
        transformation, sig = X
#        deformation = self.locnet.call(X)
#        Y = tf.expand_dims(X[...,0], 4) # only transform the first channel
        output = self._transform(transformation, sig, self.output_size) 
        return output

    def _repeat(self, x, num_repeats): # copy along the second dimension, each row is a copy of an index
        ones = tf.ones((1, num_repeats), dtype='int32')
        x = tf.reshape(x, shape=(-1,1))
        x = tf.matmul(x, ones)
        return tf.reshape(x, [-1])

    def _interpolate(self, signal, x, output_size): 

        batch_size = tf.shape(signal)[0]
#        print(tf.keras.backend.int_shape(signal))
        t_len = tf.shape(signal)[1]
        num_channels = tf.shape(signal)[-1]

        x = tf.cast(x , dtype='float32')
        scale = tf.cast(output_size[0], dtype='float32')
        
        x = x * scale

        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        
        max_x = tf.cast(t_len - 1,  dtype='int32')        
        zero = tf.zeros([], dtype='int32')

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)

        pts_batch = tf.range(batch_size*num_channels)*t_len
        flat_output_dimensions = output_size[0]
        base = self._repeat(pts_batch, flat_output_dimensions)
        
#        print(base.shape)
#        print(x0.shape)
        ind_0 = base + x0
        ind_1 = base + x1

        flat_signal = tf.transpose(signal, (0,2,1))
        flat_signal = tf.reshape(flat_signal, [-1] )

#        flat_signal = tf.reshape(signal, [-1, num_channels] )
        flat_signal = tf.cast(flat_signal, dtype='float32')
        
       
        pts_values_0 = tf.gather(flat_signal, ind_0)
        pts_values_1 = tf.gather(flat_signal, ind_1)
        
        
        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')

        w_0 = x1 - x
        w_1 = x - x0
#        w_0 = tf.expand_dims(x1 - x, 1)
#        w_1 = tf.expand_dims(x - x0, 1)

        output = w_0*pts_values_0 + w_1*pts_values_1
        
        output = tf.reshape(output, (-1, output_size[1], output_size[0]))
          
        output = tf.transpose(output, (0,2,1))
        
        return output

    def _meshgrid(self, t_length):
        x_linspace = tf.linspace(0., 1.0, t_length)
        ones = tf.ones_like(x_linspace)
        indices_grid = tf.concat([x_linspace, ones], axis=0)
#        return tf.reshape(indices_grid, [-1])
        return indices_grid

    def _transform(self, affine_transformation, input_sig, output_size):
        batch_size = tf.shape(input_sig)[0]
        t_len = output_size[0]
        num_channels = tf.shape(input_sig)[-1]
              
        indices_grid = self._meshgrid(t_len)


        indices_grid = tf.tile(indices_grid, tf.stack([batch_size*num_channels]))
        indices_grid = tf.reshape(indices_grid, (batch_size, num_channels, 2, -1) )
#
#        indices_grid = tf.tile(indices_grid, tf.stack([batch_size]))
#        indices_grid = tf.reshape(indices_grid, (batch_size, 2, -1) )
        
#        affine_transformation = tf.concat([0.5*tf.ones([batch_size, 1]), 100*tf.ones([batch_size,1])], axis = 1)
        
        affine_transformation = tf.reshape(affine_transformation, (-1, num_channels, 1, 2)) # this line is necessary for tf.matmul to perform
        affine_transformation = tf.cast(affine_transformation, 'float32')
               
#        print(indices_grid.shape)
#        print(affine_transformation.shape)
        transformed_grid = tf.matmul(affine_transformation, indices_grid) 
#        transformed_grid = indices_grid[:,0,:]
        
        x_s_flatten = tf.reshape(transformed_grid, [-1])

        transformed_vol = self._interpolate(input_sig, 
                                                x_s_flatten,
                                                output_size)

        
        return transformed_vol     


class STN_1D_noweights(Layer):
    '''
    1D spatial transformer
    '''

    def __init__(self,
#                 localization_net, # this suppose to produce a deformation with 3 channels
                 output_size,
                 **kwargs):
#        self.locnet = localization_net
        self.output_size = output_size
        super(STN_1D_noweights, self).__init__(**kwargs)

    def build(self, input_shape):
#        self.locnet.build(input_shape)
#        self.trainable_weights = self.locnet.trainable_weights
        super(STN_1D_noweights, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        output_size = self.output_size
        return (None,
                int(output_size[0]), # time
                int(output_size[1]), # channels
                )  

    def call(self, X, mask=None): 
        transformation, sig = X
#        deformation = self.locnet.call(X)
#        Y = tf.expand_dims(X[...,0], 4) # only transform the first channel
        output = self._transform(transformation, sig, self.output_size) 
        return output

    def _repeat(self, x, num_repeats): # copy along the second dimension, each row is a copy of an index
        ones = tf.ones((1, num_repeats), dtype='int32')
        x = tf.reshape(x, shape=(-1,1))
        x = tf.matmul(x, ones)
        return tf.reshape(x, [-1])

    def _interpolate(self, signal, x, output_size): 

        batch_size = tf.shape(signal)[0]
#        print(tf.keras.backend.int_shape(signal))
        t_len = tf.shape(signal)[1]
        num_channels = tf.shape(signal)[-1]

        x = tf.cast(x , dtype='float32')
        scale = tf.cast(output_size[0], dtype='float32')
        
        x = x * scale

        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        
        max_x = tf.cast(t_len - 1,  dtype='int32')        
        zero = tf.zeros([], dtype='int32')

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)

        pts_batch = tf.range(batch_size)*t_len
        flat_output_dimensions = output_size[0]
        base = self._repeat(pts_batch, flat_output_dimensions)
        
#        print(base.shape)
#        print(x0.shape)
        ind_0 = base + x0
        ind_1 = base + x1

#        flat_signal = tf.transpose(signal, (0,2,1))
        flat_signal = tf.reshape(signal, [-1, num_channels] )
        flat_signal = tf.cast(flat_signal, dtype='float32')
        
       
        pts_values_0 = tf.gather(flat_signal, ind_0)
        pts_values_1 = tf.gather(flat_signal, ind_1)
        
        
        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')

        
        w_0 = tf.expand_dims(x1 - x, 1)
        w_1 = tf.expand_dims(x - x0, 1)

        output = w_0*pts_values_0 + w_1*pts_values_1
        
        output = tf.reshape(output, (-1, output_size[0], output_size[1]))
          
     
        return output

    def _meshgrid(self, t_length):
        x_linspace = tf.linspace(0., 1.0, t_length)
        ones = tf.ones_like(x_linspace)
        indices_grid = tf.concat([x_linspace, ones], axis=0)
#        return tf.reshape(indices_grid, [-1])
        return indices_grid

    def _transform(self, affine_transformation, input_sig, output_size):
        batch_size = tf.shape(input_sig)[0]
        t_len = output_size[0]
#        num_channels = tf.shape(input_sig)[-1]
              
        indices_grid = self._meshgrid(t_len)

        indices_grid = tf.tile(indices_grid, tf.stack([batch_size]))
        indices_grid = tf.reshape(indices_grid, (batch_size, 2, -1) )
        
#        affine_transformation = tf.concat([0.5*tf.ones([batch_size, 1]), 100*tf.ones([batch_size,1])], axis = 1)
        
        affine_transformation = tf.reshape(affine_transformation, (-1, 1, 2)) # this line is necessary for tf.matmul to perform
        affine_transformation = tf.cast(affine_transformation, 'float32')
               
#        print(indices_grid.shape)
#        print(affine_transformation.shape)
        transformed_grid = tf.matmul(affine_transformation, indices_grid) 
#        transformed_grid = indices_grid[:,0,:]
        
        x_s_flatten = tf.reshape(transformed_grid, [-1])

        transformed_vol = self._interpolate(input_sig, 
                                                x_s_flatten,
                                                output_size)

        
        return transformed_vol     


class STN_1D(Layer):
    '''
    1D spatial transformer
    '''

    def __init__(self,
                 localization_net, # this suppose to produce a deformation with 3 channels
                 output_size,
                 **kwargs):
        self.locnet = localization_net
        self.output_size = output_size
        super(STN_1D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.locnet.build(input_shape)
        self.trainable_weights = self.locnet.trainable_weights
        super(STN_1D, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        output_size = self.output_size
        return (None,
                int(output_size[0]), # time
                int(output_size[1]), # channels
                )  

    def call(self, X, mask=None): 
        deformation = self.locnet.call(X)
#        Y = tf.expand_dims(X[...,0], 4) # only transform the first channel
        output = self._transform(deformation, X, self.output_size) 
        return output

    def _repeat(self, x, num_repeats): # copy along the second dimension, each row is a copy of an index
        ones = tf.ones((1, num_repeats), dtype='int32')
        x = tf.reshape(x, shape=(-1,1))
        x = tf.matmul(x, ones)
        return tf.reshape(x, [-1])

    def _interpolate(self, signal, x, output_size): 

        batch_size = tf.shape(signal)[0]
#        print(tf.keras.backend.int_shape(signal))
        t_len = tf.shape(signal)[1]
        num_channels = tf.shape(signal)[-1]

        x = tf.cast(x , dtype='float32')
        scale = tf.cast(output_size[0]-1, dtype='float32')
        
        x = x * scale

        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        
        max_x = tf.cast(t_len - 1,  dtype='int32')        
        zero = tf.zeros([], dtype='int32')

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)

        pts_batch = tf.range(batch_size)*t_len
        flat_output_dimensions = output_size[0]
        base = self._repeat(pts_batch, flat_output_dimensions)
        
#        print(base.shape)
#        print(x0.shape)
        ind_0 = base + x0
        ind_1 = base + x1

#        flat_signal = tf.transpose(signal, (0,2,1))
        flat_signal = tf.reshape(signal, [-1, num_channels] )
        flat_signal = tf.cast(flat_signal, dtype='float32')
        
       
        pts_values_0 = tf.gather(flat_signal, ind_0)
        pts_values_1 = tf.gather(flat_signal, ind_1)
        
        
        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')

        
        w_0 = tf.expand_dims(x1 - x, 1)
        w_1 = tf.expand_dims(x - x0, 1)

        output = w_0*pts_values_0 + w_1*pts_values_1
        
        output = tf.reshape(output, (-1, output_size[0], output_size[1]))
          
     
        return output

    def _meshgrid(self, t_length):
        x_linspace = tf.linspace(0., 1.0, t_length)
        ones = tf.ones_like(x_linspace)
        indices_grid = tf.concat([x_linspace, ones], axis=0)
#        return tf.reshape(indices_grid, [-1])
        return indices_grid

    def _transform(self, affine_transformation, input_sig, output_size):
        batch_size = tf.shape(input_sig)[0]
        t_len = output_size[0]
#        num_channels = tf.shape(input_sig)[-1]
              
        indices_grid = self._meshgrid(t_len)

        indices_grid = tf.tile(indices_grid, tf.stack([batch_size]))
        indices_grid = tf.reshape(indices_grid, (batch_size, 2, -1) )
        
#        affine_transformation = tf.concat([0.5*tf.ones([batch_size, 1]), 100*tf.ones([batch_size,1])], axis = 1)
        
        affine_transformation = tf.reshape(affine_transformation, (-1, 1, 2)) # this line is necessary for tf.matmul to perform
        affine_transformation = tf.cast(affine_transformation, 'float32')
               
#        print(indices_grid.shape)
#        print(affine_transformation.shape)
        transformed_grid = tf.matmul(affine_transformation, indices_grid) 
#        transformed_grid = indices_grid[:,0,:]
        
        x_s_flatten = tf.reshape(transformed_grid, [-1])

        transformed_vol = self._interpolate(input_sig, 
                                                x_s_flatten,
                                                output_size)

        
        return transformed_vol    

class SpatialTransformer(Layer):
    """Spatial Transformer Layer
    Implements a spatial transformer layer as described in [1]_.
    Borrowed from [2]_:
    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py
    .. [3]  https://github.com/EderSantana/seya/blob/keras1/seya/layers/attention.py
    """

    def __init__(self,
                 localization_net,
                 output_size,
                 **kwargs):
        self.locnet = localization_net
        self.output_size = output_size
        super(SpatialTransformer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.locnet.build(input_shape)
        self.trainable_weights = self.locnet.trainable_weights
        super(SpatialTransformer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        output_size = self.output_size
        return (None,
                int(output_size[0]),
                int(output_size[1]) )
#                int(input_shape[-1]))

    def call(self, X, mask=None):
        affine_transformation = self.locnet.call(X)
#        Y = tf.expand_dims(X[:,:,:,0], 3)
        output = self._transform(affine_transformation, X, self.output_size)
        return output

    def _repeat(self, x, num_repeats):
        ones = tf.ones((1, num_repeats), dtype='int32')
        x = tf.reshape(x, shape=(-1,1))
        x = tf.matmul(x, ones)
        return tf.reshape(x, [-1])

    def _interpolate(self, image, x, y, output_size):
        batch_size = tf.shape(image)[0]
        height = tf.shape(image)[1]
        width = tf.shape(image)[2]
#        num_channels = tf.shape(image)[3]

        x = tf.cast(x , dtype='float32')
        y = tf.cast(y , dtype='float32')

#        height_float = tf.cast(height, dtype='float32')
#        width_float = tf.cast(width, dtype='float32')

        output_height = output_size[0]
        output_width  = output_size[1]

#        x = .5*(x + 1.0)*(width_float-1)
#        y = .5*(y + 1.0)*(height_float-1)

        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        max_y = tf.cast(height - 1, dtype='int32')
        max_x = tf.cast(width - 1,  dtype='int32')
        zero = tf.zeros([], dtype='int32')

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        flat_image_dimensions = width*height
        pixels_batch = tf.range(batch_size)*flat_image_dimensions
        flat_output_dimensions = output_height*output_width
        base = self._repeat(pixels_batch, flat_output_dimensions)
        base_y0 = base + y0*width
        base_y1 = base + y1*width
        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1

#        flat_image = tf.reshape(image, shape=(-1, num_channels))
        flat_image = tf.reshape(image, shape=(-1,1))
        flat_image = tf.cast(flat_image, dtype='float32')
        pixel_values_a = tf.gather(flat_image, indices_a)
        pixel_values_b = tf.gather(flat_image, indices_b)
        pixel_values_c = tf.gather(flat_image, indices_c)
        pixel_values_d = tf.gather(flat_image, indices_d)

        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')
        y0 = tf.cast(y0, 'float32')
        y1 = tf.cast(y1, 'float32')

        area_a = tf.expand_dims(((x1 - x) * (y1 - y)), 1)
        area_b = tf.expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = tf.expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = tf.expand_dims(((x - x0) * (y - y0)), 1)

#        area_a = ((x1 - x) * (y1 - y))
#        area_b = ((x1 - x) * (y - y0))
#        area_c = ((x - x0) * (y1 - y))
#        area_d = ((x - x0) * (y - y0))
        
        
        output = tf.add_n([area_a*pixel_values_a,
                           area_b*pixel_values_b,
                           area_c*pixel_values_c,
                           area_d*pixel_values_d])
        return output

    def _meshgrid(self, height, width):
        x_linspace = tf.linspace(0., width - 1., width)
        y_linspace = tf.linspace(0., height- 1., height)
        x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)
        x_coordinates = tf.reshape(x_coordinates, [-1])
        y_coordinates = tf.reshape(y_coordinates, [-1])
        ones = tf.ones_like(x_coordinates)
        indices_grid = tf.concat([x_coordinates, y_coordinates, ones], 0)
        return indices_grid

    def _transform(self, affine_transformation, input_shape, output_size):
        batch_size = tf.shape(input_shape)[0]
        height = tf.shape(input_shape)[1]
        width = tf.shape(input_shape)[2]
#        num_channels = tf.shape(input_shape)[-1]

#        affine_transformation = tf.reshape(affine_transformation, shape=(batch_size,2,3))

        affine_transformation = tf.reshape(affine_transformation, (-1, 2, 3))
        affine_transformation = tf.cast(affine_transformation, 'float32')

        width = tf.cast(width, dtype='float32')
        height = tf.cast(height, dtype='float32')
        output_height = output_size[0]
        output_width = output_size[1]
        indices_grid = self._meshgrid(output_height, output_width)
#        indices_grid = tf.expand_dims(indices_grid, 0)
#        indices_grid = tf.reshape(indices_grid, [-1]) # flatten?

        indices_grid = tf.tile(indices_grid, tf.stack([batch_size]))
        indices_grid = tf.reshape(indices_grid, (batch_size, 3, -1))

        transformed_grid = tf.matmul(affine_transformation, indices_grid)
        x_s = tf.slice(transformed_grid, [0, 0, 0], [-1, 1, -1])
        y_s = tf.slice(transformed_grid, [0, 1, 0], [-1, 1, -1])
        x_s_flatten = tf.reshape(x_s, [-1])
        y_s_flatten = tf.reshape(y_s, [-1])

        transformed_image = self._interpolate(input_shape,
                                                x_s_flatten,
                                                y_s_flatten,
                                                output_size)

        transformed_image = tf.reshape(transformed_image, shape=(batch_size,
                                                                output_height,
                                                                output_width)
                                                                    )
        return transformed_image

class Window_trunc_no_weights(Layer):
    '''
    Use proposed window to truncate each input signal
    does not copy the localization net inside
    * Still got the same error: an operation has 'NONE' for gradient...
    '''
    def __init__(self,
                 output_size,
                 **kwargs):
        self.output_size = output_size
        super(Window_trunc_no_weights, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Window_trunc_no_weights, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        output_size = self.output_size
        return (None,
                int(output_size[0]), # time
                int(output_size[1]), # channels
                )  
        
    def call(self, X):
        assert isinstance(X, list)
        signal, start_pts = X
        output = self.truncate(signal, start_pts, self.output_size)
#        output = self.truncate(X, self.output_size)
        return output

    def repeat(self, x, num_repeats): # copy along the second dimension, each row is a copy of an index
        ones = tf.ones((1, num_repeats), dtype='int32')
        x = tf.reshape(x, shape=(-1,1))
        x = tf.matmul(x, ones)
        return tf.reshape(x, [-1])

    def truncate(self, signal, start_pts, output_size):
#    def truncate(self, signal, output_size):
        batchsize = tf.shape(signal)[0] 
        t_len = tf.shape(signal)[1] 
        num_channels = tf.shape(signal)[2]
        out_len = output_size[0]
##        out_len_float = tf.cast(out_len, 'float32')
        
        '''
        Double check the following block
        '''
        #################################################################
        batch_Cs = batchsize * num_channels
#        batch_Cs_float = tf.cast(batch_Cs, 'float32')
        grid = tf.tile( tf.range(out_len), [batch_Cs])
        
        max_t = t_len - out_len - 1       
        zero = tf.zeros([], dtype='int32')
        
        t_len_float = tf.cast(t_len, 'float32')
#        start_pts = tf.random_uniform([batchsize, num_channels], 0, 1)
        start_pts_scaled_back = start_pts * (t_len_float - 1) 
        start_pts_flat = tf.reshape(start_pts_scaled_back, [-1]) # if start_pts is between 0 and 1, i.e. sigmoid activation
#        start_pts_flat = tf.reshape(tf.floor(start_pts), [-1]) # if activation is relu in locnet's last layer
        start_pts_flat = tf.floor(start_pts_flat)
        start_pts_flat = tf.cast(start_pts_flat, 'int32')
        start_pts_flat = tf.clip_by_value(start_pts_flat, zero, max_t) # value clip here
        S_pts = self.repeat(start_pts_flat, out_len)
        

        
        gap = tf.range(batch_Cs)*t_len
        base = self.repeat(gap, out_len)
        
        indices = tf.add_n([grid , S_pts , base])
#        indices = grid + base
#        ################################################################
#        
        signal_flat = tf.transpose(signal, (0, 2, 1))
        signal_flat = tf.reshape(signal_flat, [-1] )
        
        values = tf.gather(signal_flat, indices)   
        
        values = tf.reshape(values, (-1, self.output_size[1], self.output_size[0]))
        
        values = tf.transpose(values, (0, 2, 1))

        return values    

    
class Window_trunc(Layer):
    '''
    Use proposed window to truncate each input signal
    '''
    def __init__(self,
                 localization_net, # this suppose to produce a deformation with 3 channels
                 output_size,
                 **kwargs):
        self.locnet = localization_net
        self.output_size = output_size
        super(Window_trunc, self).__init__(**kwargs)

    def build(self, input_shape):
        self.locnet.build(input_shape)
        self.trainable_weights = self.locnet.trainable_weights  # returns non-gradient error..
        super(Window_trunc, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        output_size = self.output_size
        return (None,
                int(output_size[0]), # time
                int(output_size[1]), # channels
                )  
        
    def call(self, X):
        start_pts = self.locnet.call(X)
        output = self.truncate(X, start_pts, self.output_size)
        return output

    def repeat(self, x, num_repeats): # copy along the second dimension, each row is a copy of an index
        ones = tf.ones((1, num_repeats), dtype='int32')
        x = tf.reshape(x, shape=(-1,1))
        x = tf.matmul(x, ones)
        return tf.reshape(x, [-1])

    def truncate(self, signal, start_pts, output_size):
        batchsize = tf.shape(signal)[0] 
        t_len = tf.shape(signal)[1] 
        num_channels = tf.shape(signal)[2]
        out_len = output_size[0]
#        out_len_float = tf.cast(out_len, 'float32')
        
        '''
        Double check the following block
        '''
        #################################################################
        batch_Cs = batchsize*num_channels
#        batch_Cs_float = tf.cast(batch_Cs, 'float32')
        grid = tf.tile( tf.range(out_len), [batch_Cs])

        max_t = t_len - out_len - 1       
        zero = tf.zeros([], dtype='int32')
        
        t_len_float = tf.cast(t_len, 'float32')
        start_pts_scaled_back = tf.floor(start_pts * (t_len_float - 1.0) )
        start_pts_flat = tf.reshape(start_pts_scaled_back, [-1]) # if start_pts is between 0 and 1, i.e. sigmoid activation
#        start_pts_flat = tf.reshape(tf.floor(start_pts), [-1]) # if activation is relu in locnet's last layer
        
#        start_pts_flat = tf.cast(start_pts_flat, 'int32')
#        start_pts_flat = tf.clip_by_value(start_pts_flat, zero, max_t) # value clip here
#        S_pts = self.repeat(start_pts_flat, out_len) # this block may be the problem                
        
        base = self.repeat(tf.range(batch_Cs)*t_len, out_len)

#        indices = tf.add_n([grid , S_pts , base])
        
        St = tf.transpose( tf.reshape(grid, (batch_Cs, out_len)), (1, 0) ) + start_pts_flat
        
        S_pts = tf. reshape( tf.transpose(St, (1, 0)), [-1])
        
        indices = S_pts + base

        ################################################################
        
        signal_flat = tf.transpose(signal, (0, 2, 1))
        signal_flat = tf.reshape(signal_flat, [-1] )
        
        values = tf.gather(signal_flat, indices)   
        
        values = tf.reshape(values, (-1, self.output_size[1], self.output_size[0]))
        
        values = tf.transpose(values, (0, 2, 1))
        
        return values
    


class mask(Layer):
    '''
    create boolean mask layer to mask activation to 0 and 1
    '''

    def __init__(self, thres, **kwargs):
        
        self.thres = thres
        super(mask, self).__init__(**kwargs)

    def build(self, input_shape):

        super(mask, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return shape_a

    def call(self, X): 
        assert isinstance(X, list)
        signal, attention = X
        cond = tf.greater(attention, tf.ones(tf.shape(attention))*self.thres)
        mask = tf.where(cond, tf.ones(tf.shape(attention)), tf.zeros(tf.shape(attention)))
        
        return tf.multiply(signal, mask)
#        return mask
        
class band_mask(Layer):
    '''
    Select frequencies from input with "learned" mask
    '''
    def __init__(self, thres, mask_type, **kwargs):
        
        self.thres = thres
        self.mask_type = mask_type
        super(band_mask, self).__init__(**kwargs)

    def build(self, input_shape):

        super(band_mask, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return shape_a

    def call(self, X): 
        assert isinstance(X, list)
        signal, attention = X
        signal_fft = tf.signal.rfft(tf.transpose(tf.cast(signal, 'float32'), (0,2,1)))
        
        if self.mask_type == 'hard':
            cond = tf.greater(attention, tf.ones(tf.shape(attention))*self.thres)
            mask = tf.where(cond, tf.ones(tf.shape(attention)), tf.zeros(tf.shape(attention)))
            
            signal_fft_masked = tf.multiply(signal_fft, tf.transpose(tf.cast(mask,'complex64'), (0,2,1)) ) 

        else:
            signal_fft_masked = tf.multiply(signal_fft, tf.transpose(tf.cast(attention,'complex64'), (0,2,1)) ) 
        
#        PE = tf.reduce_sum(signal_fft**2, axis=-1, keepdims = True)
#        PE_masked = tf.reduce_sum(signal_fft_masked**2, axis=-1, keepdims = True)
#        
#        signal_rec = tf.signal.irfft(signal_fft_masked * PE / PE_masked) 
        
        signal_rec = tf.signal.irfft(signal_fft_masked)
        
        return tf.transpose(signal_rec, (0, 2, 1))   
    
class F_series(Layer):
    def __init__(self, interval_length, num_components, **kwargs):
        self.L = interval_length
        self.N = num_components
        super(F_series, self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(F_series, self).build(input_shape)
        
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_coef, shape_x = input_shape
        return shape_x
    
    def call(self, X):
        assert isinstance(X, list)
        coef, x = X
        angles = tf.range(1, 1+self.N)
        angles = tf.cast(angles, 'float32')
        Pi = tf.constant(3.1415926, 'float32')
        omega = 2.0*Pi*angles/self.L
        omega = tf.reshape(omega, (1, -1)) # (1,N)
        
        phase = tf.matmul(x, omega) # (batchsize, N)
        cos_part = coef[:,1:1+self.N]*tf.math.cos(phase)
        sin_part = coef[:,1+self.N:]*tf.math.sin(phase)
        const = coef[:,:1]
        
        return const + tf.reduce_sum(cos_part + sin_part, 1, keepdims=True)
    
class F_series_multichannel(Layer):
    '''
    For regressing R - R^C functions
    '''
    def __init__(self, interval_length, num_components, num_channels,**kwargs):
        self.L = interval_length
        self.N = num_components
        self.C = num_channels
        super(F_series_multichannel, self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(F_series_multichannel, self).build(input_shape)
        
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_coef, shape_x = input_shape
        return (shape_coef[0], 1, self.C) 
    
    def call(self, X):
        assert isinstance(X, list)
        coef, x = X  # coef: (batchsize, N, C)
        angles = tf.range(1, 1+self.N)
        angles = tf.cast(angles, 'float32')

        Pi = tf.constant(3.1415926, 'float32')
        omega = 2.0*Pi*angles/self.L
        omega = tf.reshape(omega, (1, -1)) # (1,N)
        
        phase = tf.matmul(x, omega) # (batchsize, N)
#        phase = tf.expand_dims(phase, 2)
        phase = tf.reshape(tf.tile(phase, [1,self.C]), [-1, self.N, self.C])

        
#        cos_part = []
#        sin_part = []
#        for i in range(self.C):
#            cos_part.append(coef[:,1:1+self.N, i]*tf.math.cos(phase))
#            sin_part.append(coef[:,1+self.N:, i]*tf.math.sin(phase))
#        
#        cos_part = tf.concat(cos_part)    
#        sin_part = tf.concat(sin_part)  
        print(tf.shape(coef))
        cos_part = coef[:,1:1+self.N,:]*tf.math.cos(phase)
        sin_part = coef[:,1+self.N:,:]*tf.math.sin(phase)        
        const = coef[:,:1, :]
        
        return const + tf.reduce_sum(cos_part + sin_part, 1, keepdims=True)
    

    
import keras.backend as K
class Get_gradient(Layer):

    def __init__(self, order, return_all = False, **kwargs):
        self.order = order
        self.return_all = return_all
        
        super(Get_gradient, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)

        super(Get_gradient, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        assert isinstance(x, list)
        _out, _in = x
        grad=[K.gradients(_out, _in)]
        for ind in range(1, self.order):
            grad.append(K.gradients(grad[-1], _in))
#        if self.return_all:
#            return grad
#        
#        else: 
        return grad[-1]


    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_out, shape_in = input_shape
        
        return (shape_out[0], shape_in[1])
    

class VAE_SamplingLayer(Layer):
    def __init__(self, **kwargs):
        super(VAE_SamplingLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        assert isinstance(input_shape, list)

        super(VAE_SamplingLayer, self).build(input_shape)

    def kl_div_loss(self, z_mean, z_log_var):
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(kl_loss)

    def call(self, inputs):
        z_mean = inputs[0]
        z_log_var = inputs[1]
        batch_size = K.shape(z_mean)[0]
        latent_dim = K.int_shape(z_mean)[1]
        loss = self.kl_div_loss(z_mean, z_log_var)
        self.add_loss(loss, inputs=inputs)
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                                  stddev=1.0)
        return z_mean + K.exp(z_log_var / 2) * epsilon
    
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        
        return shape_a