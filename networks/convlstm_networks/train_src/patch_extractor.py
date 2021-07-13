from icecream import ic
import pdb
import numpy as np
import cv2
import pathlib
from pathlib import Path
import os
import deb
# from dataset_stats import DatasetStats

class PatchExtractor():
	def __init__(self, paramsTrain, ds):
		self.paramsTrain = paramsTrain
		self.dataset = ds
		self.dataSource = paramsTrain.dataSource 

		ic(self.dataSource)               
		# self.datasetStats=DatasetStats(self.dataset)

		self.conf={"t_len":self.dataset.t_len, "path": self.paramsTrain.path, "class_n":self.dataset.class_n, 'label':{}, 'seq':{}}
		self.conf['band_n']=self.dataSource.band_n
		self.label_folder=self.dataSource.label_folder

		self.debug=1

		ic(self.conf['path']/self.label_folder/"/")

		self.conf["train"]={}
		self.conf["train"]["mask"]={}
		train_test_mask_name="TrainTestMask.tif"
		self.conf["train"]["mask"]["dir"]=self.conf["path"]/train_test_mask_name


		self.conf["in_npy_path"]=self.conf["path"]/self.dataSource.foldernameInput
		ic(self.conf["in_npy_path"])

		self.conf["im_size"]=(self.dataset.im_h,self.dataset.im_w)
		self.conf["im_3d_size"]=self.conf["im_size"]+(self.conf["band_n"],)

		
		self.conf['center_pixel']=int(np.around(self.paramsTrain.patch_len/2))

		ic(self.conf["train"]["mask"]["dir"])
		ic(os.getcwd())

		self.mask=cv2.imread(
			str(self.conf["train"]["mask"]["dir"]),-1).astype(np.uint8)

	def label_seq_mask(self,im,mask): 
		im=im.astype(np.uint8) 
		im_train=im.copy() 
		im_test=im.copy() 
		 
		mask_train=mask.copy() 
		mask_train[mask==2]=0 
		mask_test=mask.copy() 
		mask_test[mask==1]=0 
		mask_test[mask==2]=1 
	 
		deb.prints(im.shape) 
		deb.prints(mask_train.shape) 
	 
		deb.prints(im.dtype) 
		deb.prints(mask_train.dtype) 
		 
		for t_step in range(im.shape[0]): #Iterate over time steps
			im_train[t_step]=cv2.bitwise_and(im[t_step],im[t_step],mask=mask_train) 
			im_test[t_step]=cv2.bitwise_and(im[t_step],im[t_step],mask=mask_test) 
	 
	 
		#im_train[t_step,:,:,band][mask!=1]=-1 
		#im_test[t_step,:,:,band][mask!=2]=-1 
		deb.prints(im_train.shape) 
		return im_train,im_test 

	def getFullIms(self):
		patch={}

#		pdb.set_trace()

#		ic(os.listdir(self.conf["train"]["mask"]["dir"]))
		


		patch["full_ims"]=np.zeros((self.conf["t_len"],)+self.conf["im_3d_size"]).astype(np.float16)
		self.full_label = np.zeros((self.conf["t_len"],)+self.conf["im_3d_size"][0:2]).astype(np.int8)
		patch["full_label_ims"] = self.full_label


		ic(patch["full_ims"].shape)
		ic(self.dataset.im_list)

		#=======================LOAD, NORMALIZE AND MASK FULL IMAGES ================#
		add_id=0
		patch=self.dataset.im_load(patch,self.dataset.im_list,
			self.dataset.label_list,add_id,self.conf) # replace patch[full_ims] for self.full_ims

		patch["full_ims"] = self.dataSource.clip_undesired_values(
			patch["full_ims"])
		
		print(np.min(patch["full_ims"]),np.max(patch["full_ims"]),np.average(patch["full_ims"]))

		deb.prints(self.dataset.name)
		deb.prints(self.dataset.scaler_name)		
		deb.prints(self.dataset.seq_mode)
		deb.prints(self.dataset.seq_date)
		deb.prints(self.dataset.scaler_load)


		patch["full_ims"]=self.dataSource.im_seq_normalize_hwt(patch["full_ims"],self.mask,
				scaler_load=self.dataset.scaler_load, ds_name=self.dataset.scaler_name, 
				seq_mode=self.dataset.seq_mode, seq_date=self.dataset.seq_date)

		self.full_ims_train,self.full_ims_test = (patch["full_ims"], patch["full_ims"]) # chantge to just self.full_ims. No patch
		patch["full_ims"]=[]

		
		self.full_label_train,self.full_label_test=self.label_seq_mask(
			self.full_label,self.mask)  #use self.full_label instead of patch


		unique,count=np.unique(self.full_label_train,return_counts=True) 
		print("Train masked unique/count",unique,count) 
		unique,count=np.unique(self.full_label_test,return_counts=True) 
		print("Test masked unique/count",unique,count) 

		ic(self.paramsTrain.path / 'full_ims/full_ims_test.npy')

		np.save(self.paramsTrain.path / 'full_ims/full_ims_test.npy',self.full_ims_test.astype(np.float16))
		np.save(self.paramsTrain.path / 'full_ims/full_ims_train.npy',self.full_ims_train.astype(np.float16))
		np.save(self.paramsTrain.path / 'full_ims/full_label_test.npy',self.full_label_test)
		np.save(self.paramsTrain.path / 'full_ims/full_label_train.npy',self.full_label_train)


#		np.save(self.paramsTrain.path / 'full_ims/full_label.npy',self.full_label)

	def fullImsLoad(self):
		self.full_ims_test = np.load(self.paramsTrain.path / 'full_ims/full_ims_test.npy')
		self.full_ims_train = np.load(self.paramsTrain.path / 'full_ims/full_ims_train.npy')
		self.full_label_test = np.load(self.paramsTrain.path / 'full_ims/full_label_test.npy')
		self.full_label_train = np.load(self.paramsTrain.path / 'full_ims/full_label_train.npy')

		


	# Is this correct??
	def is_mask_from_train(self,mask_patch,label_patch): 
		condition_1=(mask_patch[self.conf["center_pixel"],self.conf["center_pixel"]]==1)
		condition_2=(label_patch[self.conf["center_pixel"],self.conf["center_pixel"]]>0)
		return (condition_1 and condition_2)

	def extract(self):
		print("STARTED PATCH EXTRACTION")

		self.full_label = self.full_ims_train + self.full_ims_test

		t_steps, h, w, channels = self.full_ims_train.shape

		mask_train=np.zeros((h,w))
		mask_test=np.zeros((h,w))

		window = self.paramsTrain.patch_len
		stride = self.paramsTrain.stride # patch_len: 32

		gridx = range(window//2, w - window//2, stride)
		gridx = np.hstack((gridx, w - window//2))

		gridy = range(window//2, h - window//2, stride)
#		deb.prints(len(gridy))
		gridy = np.hstack((gridy, h - window//2))

		deb.prints(gridx.shape)
		deb.prints(gridy.shape)


		# ==================== PATCH LOCATIONS FOR RECONSTRUCTION====#

		coords_train = []
		coords_test = []

		counter=0

		#======================== START IMG LOOP ==================================#
		for i in range(len(gridx)):
			for j in range(len(gridy)):
				counter=counter+1
				if counter % 10000000 == 0:
					deb.prints(counter,fname)
				xx = gridx[i]
				yy = gridy[j]

				indexes = (yy, xx)
				bounds_y = (yy, yy + window)
				bounds_x = (xx, xx + window)


				label_patch = self.full_label[:,bounds_y[0]: bounds_y[1], bounds_x[0]: bounds_x[1]]
				mask_patch = self.mask[bounds_y[0]: bounds_y[1], bounds_x[0]: bounds_x[1]]

				if np.all(label_patch==0):
					continue
				

				if np.any(mask_patch==1): # Train sample
					coords_train.append(indexes)
				elif np.any(mask_patch==2): # Test sample
					coords_test.append(indexes)
				
		coords_train = np.asarray(coords_train)
		coords_test = np.asarray(coords_test)
		ic(coords_train.shape, coords_test.shape)
		ic(coords_train.dtype)
		ic(coords_train[0])

		np.save(self.paramsTrain.path / "coords_train.npy", coords_train) # to-do: create coords folder
		np.save(self.paramsTrain.path / "coords_test.npy", coords_test)

		#pdb.set_trace()

