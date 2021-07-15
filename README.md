# This is the code for the paper "Open Set + Fully Convolutional Recurrent Networks for Multidate Crop Recognition from Multitemporal Image Sequence"


## Installing the required python packages

The list of anaconda commands to recreate the environment for this project is in requirements.txt

## Preparing the input images 

Download the input images from the following links. 


The dataset structure is as follows. Place the sequence of NPY input images in the in_np2/ folder, and the sequence of TIF labels in the labels/ folder.

dataset/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;dataset/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{dataset_folder}/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;in_np2/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;labels/  
  
  Where dataset_folder is cv_data for Campo Verde and lm_data for LEM
  
  
## Instructions

The main routine is in main.py
