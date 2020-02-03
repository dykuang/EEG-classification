# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 15:29:41 2019

@author: dykua

Some preprocess functions including functions that load data to numpy array.
"""

from scipy.io.arff import loadarff
import numpy as np

'''
Load .arff file
'''
def get_array_from_data(data):
    arr=[]
    cate=[]
    for ind in range(len(data)):
        L = list(data[ind])
        arr.append(L[:-1]) # will need to double check the index range
        cate.append(L[-1])
    return arr, cate

def load_arff(filepath):
    '''
    Load .arff files into numpy array
    input: path for .arff file
    output: X, data part in numpy array
            Y, info part in numpy array
    '''
    data, meta = loadarff(filepath)
    X, Y = get_array_from_data(data) ## X, Y
    X = np.array(X)
    Y = np.array(Y)
    
    return X, Y

from sklearn import preprocessing
def str_to_num_label(Y):
    '''
    convert categorical labels like 'a', 'b', ... to numerical lables 0, 1, 2
    input: label array Y: (samples, )
    output: num_label, the converted label
            le, the transformer fitted to Y
    
    If need to transform labels 'Y_test' in test set, just run 'le.transform(Y_test)'.
    
    '''
    le = preprocessing.LabelEncoder()
    le.fit(Y)
    num_label = le.transform(Y)
    
    return num_label, le

def smooth1D(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    half_window_len = int(window_len/2)
    return y[half_window_len:-half_window_len]

def normalize_samples(X_train, X_test, func, *args):
    '''
    Normalize time series per channel in dataset
    Input:
        X: (samples, time, channels)
        scaler: sklearn scaler objects such as MinMaxScaler, ...
    '''
    assert len(X_train.shape)==3, "wrong input shape"

    S,T,C = X_train.shape
    scaler = []
    X_transformed = np.empty((S,T,C))
    
    if X_test is not None:
        X_test_transformed = np.empty(X_test.shape)
        for i in range(C):
            scaler.append(func(args))
            X_transformed[...,i] = scaler[i].fit_transform(X_train[...,i])
            X_test_transformed[...,i] = scaler[i].transform(X_test[...,i])
        
        return X_transformed, X_test_transformed, scaler
    else:
        for i in range(C):
            scaler.append(func(*args))
            X_transformed[...,i] = scaler[i].fit_transform(X_train[...,i])
            
        return X_transformed, scaler
    
def normalize_mvar(Xtrain, Xtest):
    assert len(Xtrain.shape)==3, "wrong input shape"
    S,T,C = Xtrain.shape
    X_transformed = np.empty((S,T,C))
    X_test_transformed = np.empty(Xtest.shape)
    for i in range(C):
        Max = np.amax(Xtrain[:,:,i]) 
        Min = np.amin(Xtrain[:,:,i]) 
        X_transformed[:,:,i] = (Xtrain[:,:,i] - Min) / (Max - Min)
        X_test_transformed[...,i] = (Xtest[:,:,i] - Min) / (Max - Min)
    
    return X_transformed, X_test_transformed

    
'''
rolling_max feature
'''
import pandas as pd  
def rolling_max(X, window = 10):
    '''
    X: (samples, time, channels)
    '''
    assert len(X.shape)==3, "wrong input shape"
    S, T, C = X.shape
    Mx = np.empty(X.shape)
    for i in range(S):
        for j in range(C):

            df = pd.DataFrame(X[i,:,j])
            Mx[i,window-1:,j] = df.rolling(window).max()[0][window-1:]
    
    return Mx
    
def rolling_mean(X, window = 10):
    '''
    X: (samples, time, channels)
    '''
    assert len(X.shape)==3, "wrong input shape"
    S, T, C = X.shape
    Mx = np.empty(X.shape)
    for i in range(S):
        for j in range(C):

            df = pd.DataFrame(X[i,:,j])
            Mx[i,window-1:,j] = df.rolling(window).mean()[0][window-1:]
    
    return Mx    


'''
Apply smoothing.
Scipy.singal or Pandas rolling_mean, rolling_max.
'''
import scipy.signal as signal

def Buterworth(X, cut_off_freq = 0.2, order = 3):
    
    # First, design the Buterworth filter
#    N  = 3    # Filter order
#    Wn = 0.1 # Cutoff frequency
    B, A = signal.butter(order, cut_off_freq, output='ba')
    smooth_data = signal.filtfilt(B,A, X)
    
    return smooth_data

def normalize_in_time(X, smooth = True, cut_off_freq = 0.2, order = 3):
    '''
    Normalize time series per channel per sample
    Input:
        X: (samples, time, channels)
        scaler: sklearn scaler objects such as MinMaxScaler, ...
    '''
    assert len(X.shape)==3, "wrong input shape"

    S,T,C = X.shape
    X_transformed = np.empty((S,T,C))
    Max = np.max(X, axis=1)
    Min = np.min(X, axis=1)
    Range = Max - Min
    if smooth:
        for i in range(S):
            for j in range(C):
                X_transformed[i,:,j] = Buterworth( (X[i,:,j] - Min[i,j])/Range[i,j], 
                                                   cut_off_freq = cut_off_freq, order = order)
    else:
        for i in range(S):
            for j in range(C):
                X_transformed[i,:,j] =  (X[i,:,j] - Min[i,j])/Range[i,j]
                                                   
    
    return X_transformed


def Buterworth_batch(X, cut_off_freq = 0.2, order = 3):
    '''
    filter the whole data batch X: (samples, time, channels)
    '''
    assert len(X.shape) == 3, "wrong input shape"
    S, T, C = X.shape
    X_smoothed = np.empty(X.shape)
    for i in range(S):
        for j in range(C):
            X_smoothed[i,:,j] = Buterworth(X[i,:,j], cut_off_freq, order)
    
    return X_smoothed
                       
from scipy.signal import periodogram, welch
def power_spectrum_batch(X, freq, use_welch=True, nseg=128):
    '''
    Get the powerspectrum of X: (samples, time, channels)
    '''   
    assert len(X.shape) == 3, "wrong input shape"
    S, T, C = X.shape  
#    Ps = np.empty((S,T//2+1, C))
    if use_welch:
        _, Ps = welch(X, freq,  detrend='constant', scaling='density', nperseg=nseg, axis = 1)
        
    else:
                         
        _, Ps = periodogram(X, freq,  detrend='constant', scaling='density', axis = 1)
    
    return Ps**0.5


def get_epdf(X, span, bins):
    '''
    Convert to emprical probability density esitimation of input X
    '''
    assert len(X.shape) == 3, "wrong input shape"
    S, T, C = X.shape
    density = np.empty([S, bins, C])
    for i in range(S):
        for j in range(C):
            density[i,:,j] = np.histogram(X[i,:,j], bins, range=span, density=True)[0]
    
    return density



def band_decompose(X, fs=400, bands={'Delta': (0, 4),
                                     'Theta': (4, 8),
                                     'Alpha': (8, 12),
                                     'Beta': (12, 30),
                                     'Gamma': (30, 45)}):
    '''
    extract relative band energy from signal `X` according to `bands`
    X: (batchsize, samples, channels)
    fs: sampling frequency
    bands: a dict of frequecy bands
    '''
    # Get real amplitudes of FFT (only in postive frequencies)
    fft_vals = np.absolute(np.fft.rfft(X, axis = 1))
    
    total_energy = np.sum(fft_vals**2, axis=1)
    # Get frequencies for amplitudes in Hz
    fft_freq = np.fft.rfftfreq(X.shape[1], 1.0/fs)
    band_power = []
    for band in bands:  
        freq_ix = np.where((fft_freq >= bands[band][0]) & 
                           (fft_freq <= bands[band][1]))[0]
        
#        band_power.append(np.median(fft_vals[:,freq_ix,:], axis = 1)) # what metric to use here?
        band_power.append(np.sum(fft_vals[:,freq_ix,:]**2, axis = 1)/total_energy)
    
    return np.stack(band_power, axis=1) # normalize?

from sklearn.metrics.pairwise import cosine_similarity as cos
def connect_matrix(X, win_len, skip=1):
    '''
    convert the (batch, seq, channels) input to (batch, seq, connect matrix) representation
    win_len: length of moving window
    skip: the skip of the moving window
    '''
    batchsize, nsamples, nchannels = X.shape 
    nsplits = (nsamples-win_len)//skip +1
    output = np.empty([batchsize, nsplits, nchannels, nchannels])
    for i in range(nsplits):
        trunc = X[:,i*skip:(i*skip+win_len),:].transpose(0, 2, 1)
        for j in range(batchsize):
            output[j,i,...] = cos(trunc[j], trunc[j])
    
    return output
    
def get_lower_info(CM):
    '''
    Get the lower left part of the dynamic connectivity matrix representation as a time series
    '''
    batchsize, steps, height, width = CM.shape
    feature_dim = (height*width - height)//2
    output = np.empty([batchsize, steps, feature_dim])
    
    for b in range(batchsize):
        for s in range(steps):
            count = 0
            for i in range(1,height):
                for j in range(i):
                    output[b,s,count] = CM[b,s,i,j]
                    count+=1
                    
    return output

'''
Convert time series from numpy format to pandas dataframe suitable for tsfresh
'''

def to_tsfresh_df(X):
    '''
    input: X, (samples, time, channels)
    output: pandas data frame, tsfresh input format
    '''
    S, T, C = X.shape
    X_flat = X.reshape((-1, C))
    cols=['C{}'.format(i+1) for i in range(C)]
    X_DF = pd.DataFrame(data=X_flat,    
                        columns=cols) 
    
    ID = np.repeat(np.arange(S), T)
    X_DF['Sample_ID'] = ID
    time = np.tile(np.arange(T), S)
    X_DF['time'] = time
    
    return X_DF


'''
bandpass filter, copied from 
https://users.soe.ucsc.edu/~karplus/bme51/w17/bandpass-filter.py
''' 
   
def band_pass(values, low_end_cutoff, high_end_cutoff, sampling_freq):
    # The band-pass filter will pass signals with frequencies between
    # low_end_cutoff and high_end_cutoff
    lo_end_over_Nyquist = low_end_cutoff/(0.5*sampling_freq)
    hi_end_over_Nyquist = high_end_cutoff/(0.5*sampling_freq)
    
    # If the bandpass filter gets ridiculously large output values (1E6 or more),
    # the problem is numerical instability of the filter (probably from using a
    # high sampling rate).  
    # The problem can be addressed by reducing the order of the filter (first argument) from 5 to 2.
    bess_b,bess_a = signal.iirfilter(5,
                Wn=[lo_end_over_Nyquist,hi_end_over_Nyquist],
                btype="bandpass", ftype='bessel')
    bandpass = signal.filtfilt(bess_b,bess_a,values)
    
# =============================================================================
#     # The low-pass filter will pass signals with frequencies
#     # below low_end_cutoff
#     bess_b,bess_a = scipy.signal.iirfilter(5, Wn=[lo_end_over_Nyquist],
#                 btype="lowpass", ftype='bessel')
#     lowpass = scipy.signal.filtfilt(bess_b,bess_a,values)
# =============================================================================
    
    return bandpass

def batch_band_pass(values, low_end_cutoff, high_end_cutoff, sampling_freq):
    assert len(values.shape) == 3, "wrong input shape"
    S, T, C = values.shape
    X_filtered = np.empty(values.shape)
    for i in range(S):
        for j in range(C):
            X_filtered[i,:,j] = band_pass(values[i,:,j], low_end_cutoff, high_end_cutoff, sampling_freq)
    
    return X_filtered

'''
Some visualization
'''
import matplotlib.pyplot as plt
def random_plot(X, Y): 
    '''
    Randomly plot sample eegs in the data
    X: (samples, time), the actual eeg values 
    Y: (samples, ), the classfication label
    '''
    n_samples, time_steps = X.shape
    selected_ind = np.random.choice(n_samples, 9, replace='False')
    plt.figure()
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.plot(X[selected_ind[i]])
        plt.title(Y[selected_ind[i]])
        
        
'''
For test purpose
'''
#if __name__ == '__main__':
#    Xtest = np.load(r'../Xtest.npy')
#    plt.plot(Xtest[0,:,0])
#    plt.plot(Buterworth(Xtest[0,:,0],cut_off_freq=0.25))





   
    
    
    