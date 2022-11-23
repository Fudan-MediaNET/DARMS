# DARMS
**DARMS is a Device-free Human Activity Recognition System based on Dual-channel Transformer Using WIFI Signals.**  



## Requirements
**MATLAB R2020a**

**PyTorch 1.10.0**



## Getting Started  
1. Click [here](https://drive.google.com/file/d/1QZXR_L3nofr4SJfpF6q2tRdTqCn7tXMP/view?usp=sharing) to download the raw CSI data and put it into the raw_data folder.  

2. Using the python/mat2npy.py to genernate the dataset.  

3. Then, you can quickly use DARMS by runing the python/DARMS_Main.py to train and validate the nerual network.  



## Project Structure
    DARMS
    ├─ matlab                 // The pre-processsing algorithm programmed with matlab
    ├─ python                 // The nerual network programmed with python
    ├─ raw_data               // The raw CSI data
    ├─ tmp                    // Storing the intermediate results.
    └─ dataset                // The dataset for training and validating the nerual network
      

## Citation
If you feel our data, codes or paper helpful in your research, please cite this work.

@article{gu2022device,
  title={Device-Free Human Activity Recognition Based on Dual-Channel Transformer Using WiFi Signals},
  author={Gu, Zhihao and He, Taiwei and Wang, Ziqi and Xu, Yuedong},
  journal={Wireless Communications and Mobile Computing},
  volume={2022},
  publisher={Hindawi}
}
