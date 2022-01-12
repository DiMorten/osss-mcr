# This is the code for the following papers:

1. Fully Convolutional Recurrent Networks for Multidate Crop Recognition From Multitemporal Image Sequences
2. "Open Set + Fully Convolutional Recurrent Networks for Multidate Crop Recognition from Multitemporal Image Sequence"


## Installing the required python packages

Environment can be installed using environment.yml file. Use the following commands:
  ```
  conda env create -f environment.yml
  conda activate tf2
  ```
## Preparing the input images 

Download the input images from the following links. 


The dataset structure is as follows. Place the sequence of NPY input images in the in_sar/ folder, and the sequence of TIF labels in the labels/ folder.
```
dataset/  
  dataset/  
    {dataset_folder}/  
      in_sar/  
      labels/  
```  
Where dataset_folder is cv_data for Campo Verde and lm_data for LEM
  
  
## Closed set training and evaluation

```
python train_and_evaluate.py
```

## Open Set training and evaluation


```
python train_and_evaluate_open_set.py
```


