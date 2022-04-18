# Classifying Functional Neuroimaging Signals

The Dual Stream Neural Network (DSNN) is an [*end-to-end*](https://towardsdatascience.com/e2e-the-every-purpose-ml-method-5d4f20dafee4) network designed to classify broad classes of functional neuroimaging signals.  

An implementation for the paper   
  [*Dual stream neural networks for brain signal classification*](https://iopscience.iop.org/article/10.1088/1741-2552/abc903/meta)  
along with some other models for possibly improving the prediction performance.
***
 
## Code Overview

* `preprocess.py`: Functions for preprocessing data.   
* `visual.py`: Some functions for visualization purpose. 
* `loss.py`: Custom losses.     
* `modules.py`: Custom network layers that will be called in `architectures.py`  
* `architectures.py`: Architectures used in training.    
* `Utils.py`: Some Utilility functions for different training style.      
* `train.py`: Training script contains flags for different types of training including:       
                   * _2D flag_: Treating the multi-channel eegs as 2d images;    
                   *_aug flag_: Training with generators. Randomly roll the signal or randomly select reconstructed signals with  part of frequencies from original.    
                   * class weights can also be manually specified.    
* `train_test_PA.py`: Using the point attention (PA) module. A second network will predict a binary mask to select a subset of sampled points from the orginal input.    
* `train_test_FS.py`: Using the band_mask module. A second network will predict a binary mask in the frequency domain and reconstructs signal to-be-classified using selected frequencies before feeding it to the downstream classifier.    
* `train_testing_resample.py`: Using the resample module. A second network will produce a new grid (not uniformed, not necessarily the same resolution) to resample the input with interpolation. The resampled signal will then flow to the classifier.  It has the option to specify whether all channels will share the same grid or using different grids.    
* `train_test_window.py`: Using the window_trunc module. A second network will predict a starting point. Input signals will be truncated from this point with fixed length before going to the classifier. (Does not compile with current implementation, still working on it.)  
* `train_DCM.py`: Using the dynamic connectivity matrix as feature, which will be mapped to a latent representation with timely distributed dense layer and then a simply RNN will be used for classification. (Huge gap between train and validation.)
* `train_w_CM.py`: Based on `train.py`, adding a paralled second branch for taking the static connectivity matrix as one extra input.      
* `train_PT_loc_tune.py`: Script for testing the performance when placing the attention module at different locations.  
* `train_clf_vae.py`: Try combining the VAE with current classifier. The current version has problems in the quality of predicted reconstruction, need a fix.  

## Current performance  
[BCI-IV dataset 3](http://www.bbci.de/competition/iv/results/index.html#dataset3):  About 10% or more accuracy improvements on both subjects. `train.py` works generall well. `train_w_CM` often outperforms it in S2. If accuracy is < 50%, replacing SVM as the last layer usually helps. But if accuracy is > 60%, this approach results in less accuracy.
