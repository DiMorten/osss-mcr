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
  
  
## Instructions for default parameters

The main routine is in main.py. For default training use:

cd src/...train_src/
python main.py

For open set training, in params_reader.py, change the parameter:

mode='OpenSet'

Then:

cd src/...train_src/
python main.py

For ROC AUC curve metrics, execute:

cd src/...analysis/
python analysis_...py

For qualitative results, execute:

cd src/...reconstruct/
python reconstruct...py

## Instructions for changing parameters

### Training parameters 
In params_reader.py, edit the next parameters:

self.dataset: 'cv' for CV dataset and 'lm' for LEM dataset
self.date: 'jun' for CV dataset and 'jun', 'mar' for LEM dataset

### AUC ROC metric and qualitative results parameters

In params_analysis.py, edit the next parameters:

self.open_mode: 'SoftmaxThresholding', 'OpenPCS'
self.covMatrixInvert: False for OpenPCS, True for OpenPCS++



