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
#        self.trainable_weights = self.locnet.trainable_weights  # returns non-gradient error..
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
        start_pts_scaled_back = tf.floor(start_pts * (t_len_float - 1) )
        start_pts_flat = tf.reshape(start_pts_scaled_back, [-1]) # if start_pts is between 0 and 1, i.e. sigmoid activation
#        start_pts_flat = tf.reshape(tf.floor(start_pts), [-1]) # if activation is relu in locnet's last layer
        
        start_pts_flat = tf.cast(start_pts_flat, 'int32')
        start_pts_flat = tf.clip_by_value(start_pts_flat, zero, max_t) # value clip here
        S_pts = self.repeat(start_pts_flat, out_len)
        
        base = self.repeat(tf.range(batch_Cs)*t_len, out_len)
        
        indices = tf.add_n([grid , S_pts , base])
#        indices = grid + base
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
    def __init__(self, thres, **kwargs):
        
        self.thres = thres
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
        cond = tf.greater(attention, tf.ones(tf.shape(attention))*self.thres)
        mask = tf.where(cond, tf.ones(tf.shape(attention)), tf.zeros(tf.shape(attention)))
        
        signal_fft = tf.signal.rfft(tf.transpose(tf.cast(signal, 'float32'), (0,2,1)))
        signal_fft_masked = tf.multiply(signal_fft, tf.transpose(tf.cast(mask,'complex64'), (0,2,1)) ) 
        signal_rec = tf.signal.irfft(signal_fft_masked)
        
        
        return tf.transpose(signal_rec, (0, 2, 1))   
