# DARMS
**DARMS is a Device-free Human Activity Recognition System based on Dual-channel Transformer Using WIFI Signals.**  



## Requirements
**MATLAB R2020a**

**PyTorch 1.10.0**



## Getting Started  
1. Click [here](https://drive.google.com/drive/folders/1uhvyzDL-A9LQRhdQdebNI9TtZqHBKQlo?usp=sharing) to download the raw data and put it into the raw_data folder.  

2. You can quickly use DARMS by running matlab/DARMS_Main.m to genernate the dataset.  

3. Then runing the python/DARMS_Main.py to train and validate the nerual network.  



## Project Structure
    DARMS
    ├─ matlab                 // The pre-processsing algorithm programmed with matlab
    ├─ python                 // The nerual network programmed with python
    ├─ raw_data               // The raw CSI data
    ├─ tmp                    // Storing the tmp file
    └─ dataset                // The dataset for training and validating the nerual network
      


