# EEG classification
 
## Code Overview

`preprocess.py`: Functions for preprocessing data.  
`visual.py`: Some functions for visualization purpose.  
`loss.py`: Custom losses.  
`modules.py`: Custom network layers that will be called in `architectures.py`
`architectures.py`: Architectures used in training.  
`Utils.py`: Some Utilility functions for different training style.    
`train.py`: Training script contains flags for different types of training including:     
                   * _2D flag_: Treating the multi-channel eegs as 2d images;  
                   *_aug flag_: Training with generators. Randomly roll the signal or randomly select reconstructed signals with  part of frequencies from original.  
                   * class weights can also be manually specified.  
`train_test_PA.py`: Using the point attention (PA) module. A second network will predict a binary mask to select a subset of sampled points from the orginal input.  
`train_test_FS.py`: Using the band_mask module. A second network will predict a binary mask in the frequency domain and reconstructs signal to-be-classified using selected frequencies before feeding it to the downstream classifier.  
`train_test_resample.py`: Using the resample module. A second network will produce a new grid (not uniformed, not necessarily the same resolution) to resample the input with interpolation. The resampled signal will then flow to the classifier.  It has the option to specify whether all channels will share the same grid or using different grids.  
`train_test_window.py`: Using the window_trunc module. A second network will predict a starting point. Input signals will be truncated from this point with fixed length before going to the classifier. (Does not compile with current implementation, still working on it.)

## Current performance  
[BCI-IV dataset 3](http://www.bbci.de/competition/iv/results/index.html#dataset3):  About 10% or more accuracy improvements on both subjects. 
