
"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import os
import math
import random

import numpy as np
from time import gmtime, strftime
import glob

import cv2
import pathlib
from pathlib import Path

import sys
import pickle
# Local
import deb
import argparse
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod
import time, datetime
import pdb
import joblib
from icecream import ic
class DataSource(object):
	def __init__(self, band_n, foldernameInput, label_folder,name):
		self.band_n = band_n
		self.foldernameInput = foldernameInput
		self.label_folder = label_folder
		self.name=name
		self.channelsToMask=range(self.band_n)
	
	@abstractmethod
	def im_load(self,filename,conf):
		pass

	def addHumidity(self):
		self.band_n=self.band_n+1

class SARSource(DataSource):

	def __init__(self):
		name='SARSource'
		band_n = 2
		foldernameInput = "in_sar/"
		label_folder = 'labels'
		super().__init__(band_n, foldernameInput, label_folder,name)


	def im_seq_normalize3(self,im,mask, scaler_load = False):
		im_check_flag=False
		t_steps,h,w,channels=im.shape

		im_flat=np.transpose(im,(1,2,3,0))

		im_flat=np.reshape(im_flat,(h*w,channels*t_steps))
		if im_check_flag==True:
			im_check=np.reshape(im_flat,(h,w,channels,t_steps))
			im_check=np.transpose(im_check,(3,0,1,2))
			deb.prints(im_check.shape)
			deb.prints(np.all(im_check==im))
		deb.prints(im.shape)
		mask_flat=np.reshape(mask,-1)

		deb.prints(im_flat[mask_flat==1,:].shape)
		print(np.min(im_flat[mask_flat==1,:]),np.max(im_flat[mask_flat==1,:]),np.average(im_flat[mask_flat==1,:]))

		scaler=StandardScaler()
		scaler_filename = 'normalization_scaler.pkl'
		if scaler_load == False:
			scaler.fit(im_flat[mask_flat==1,:])
			joblib.dump(scaler, scaler_filename)
		else:
			scaler = joblib.load(scaler_filename)  

		im_norm_flat=scaler.transform(im_flat)
		del im_flat
		im_norm=np.reshape(im_norm_flat,(h,w,channels,t_steps))
		del im_norm_flat
		deb.prints(im_norm.shape)
		im_norm=np.transpose(im_norm,(3,0,1,2))
		deb.prints(im_norm.shape)

		print("FINISHED NORMALIZING, RESULT:")
		print(np.min(im_norm),np.max(im_norm),np.average(im_norm))
		return im_norm
	def im_seq_normalize_hwt(self,im,mask, scaler_load = False, ds_name='', seq_mode='', seq_date=''):
		im_check_flag=False
		t_steps,h,w,channels=im.shape

		im_flat=np.reshape(im,(h*w*t_steps,channels))
		if im_check_flag==True:
			im_check=np.reshape(im_flat,(h,w,channels,t_steps))
			im_check=np.transpose(im_check,(3,0,1,2))
			deb.prints(im_check.shape)
			deb.prints(np.all(im_check==im))
		deb.prints(im.shape)
		# mask is (h,w). Convert to (t_step,h,w)
		mask = np.repeat(np.expand_dims(mask,axis=0),t_steps,axis=0)
		mask_flat=np.reshape(mask,-1)
		#train_flat=im_flat[mask_flat==1,:]

		deb.prints(im_flat[mask_flat==1,:].shape)
		print(np.min(im_flat[mask_flat==1,:]),np.max(im_flat[mask_flat==1,:]),np.average(im_flat[mask_flat==1,:]))

		scaler=StandardScaler()
		scaler_filename = 'normalization_scaler_'+ds_name+'_'+seq_mode+'_'+seq_date+'.pkl'
		if scaler_load == False:
			scaler.fit(im_flat[mask_flat==1,:])
			joblib.dump(scaler, scaler_filename)
		else:
			scaler = joblib.load(scaler_filename)  
			print("===== LOADING SCALER =======")
		

		im_norm_flat=scaler.transform(im_flat)
		del im_flat
		im_norm=np.reshape(im_norm_flat,(t_steps,h,w,channels))
		del im_norm_flat
		deb.prints(im_norm.shape)

		print("FINISHED NORMALIZING, RESULT:")
		print(np.min(im_norm),np.max(im_norm),np.average(im_norm))
		return im_norm

	def clip_undesired_values(self, full_ims):
		full_ims[full_ims>1]=1
		return full_ims
	def im_load(self,filename,conf):
		return np.load(filename)


class Dataset(object):
	def __init__(self,path,im_h,im_w,class_n,class_list,name,padded_dates,seq_mode,seq_date,scaler_load,scaler_name,seq_len):
		self.path=Path(path)
		self.class_n=class_n
		self.im_h=im_h
		self.im_w=im_w
		self.class_list=class_list
		self.name=name
		self.padded_dates=padded_dates
		self.seq_mode=seq_mode
		self.seq_date=seq_date
		self.scaler_load = scaler_load
		self.scaler_name = scaler_name
		self.seq_len = seq_len
	@abstractmethod
	def addDataSource(self,dataSource):
		pass
	def getBandN(self):
		return self.dataSource.band_n
	def getClassN(self):
		return self.class_n
	def getClassList(self):
		return self.class_list
	def getTimeDelta(self,delta=False,format='seconds'):
		time_delta=[]
		for im in self.im_list:
			date=im[:8]
			print(date)
			time_delta.append(time.mktime(datetime.datetime.strptime(date, 
                                             "%Y%m%d").timetuple()))
		
		time_delta = np.asarray(time_delta)
		if delta==True:
			time_delta = np.diff(time_delta)
		if format=='days':
			time_delta = time_delta/(60*60*24)
		print(time_delta)
		return time_delta
	def getDayOfTheYear(self):
		dotys = []
		dotys_sin = []
		dotys_cos = []
		for im in self.im_list:
			date=im[:8]
			print(date)
			doty = datetime.datetime.strptime(date, 
                                             "%Y%m%d").timetuple().tm_yday
			dotys.append(doty)
			doty_sin = np.sin((doty-1)*(2.*np.pi/366))
			doty_sin = (doty_sin + 1) / 2 # range [0,1]
			dotys_sin.append(doty_sin.astype(np.float16))

			doty_cos = np.cos((doty-1)*(2.*np.pi/366))
			doty_cos = (doty_cos + 1) / 2 # range [0,1]
			dotys_cos.append(doty_cos.astype(np.float16))

		dotys_sin_cos = np.concatenate((
			np.expand_dims(np.asarray(dotys_sin), axis=-1),
			np.expand_dims(np.asarray(dotys_cos), axis=-1)),
			axis=-1			
		)
		print("dotys_sin_cos.shape", dotys_sin_cos.shape)

		np.set_printoptions(suppress=True)
		print(dotys)
		print(dotys_sin_cos)
		np.set_printoptions(suppress=False)
		self.dotys_sin_cos = dotys_sin_cos
		return np.asarray(dotys), dotys_sin_cos
	def dotyReplicateSamples(self, sample_n=16):#,batch['label'].shape[0]
		self.dotys_sin_cos = np.expand_dims(self.dotys_sin_cos,axis=0) # add batch dimension
		self.dotys_sin_cos = np.repeat(self.dotys_sin_cos, sample_n, axis=0)
		deb.prints(self.dotys_sin_cos.shape)
		return self.dotys_sin_cos
	def setDotyFlag(self, doty_flag):
		self.doty_flag = doty_flag

	def addDoty(self, input_, bounds=None):
		if self.doty_flag==True:
			if bounds!=None:
				dotys_sin_cos = self.dotys_sin_cos[:,bounds[0]:bounds[1] if bounds[1]!=0 else None]
			else:
				dotys_sin_cos = self.dotys_sin_cos.copy()
			input_ = [input_, self.dotys_sin_cos]
		return input_	
	def addDotyPadded(self, input_, bounds=None, seq_len=12, sample_n=16):
		if self.doty_flag==True:
			if bounds!=None:
#				deb.prints(bounds)
				dotys_sin_cos = self.dotys_sin_cos[:,bounds[0]:bounds[1] if bounds[1]!=0 else None]
#				deb.prints(self.dotys_sin_cos.shape)
#				deb.prints(dotys_sin_cos.shape)
			else:
				dotys_sin_cos = self.dotys_sin_cos.copy()
			dotys_sin_cos_padded = np.zeros((sample_n, seq_len, 2))
			dotys_sin_cos_padded[:, -dotys_sin_cos.shape[1]:] = dotys_sin_cos
			input_ = [input_, dotys_sin_cos_padded]
		return input_
	def im_load(self,patch,im_names,label_names,add_id,conf):
		fname=sys._getframe().f_code.co_name
		for t_step in range(0,conf["t_len"]):	
			print(t_step,add_id)
			deb.prints(conf["in_npy_path"]/(im_names[t_step]+".npy"))
			#patch["full_ims"][t_step] = np.load(conf["in_npy_path"]+names[t_step]+".npy")[:,:,:2]
			patch["full_ims"][t_step] = self.dataSource.im_load(conf["in_npy_path"]/(im_names[t_step]+".npy"),conf)
			#patch["full_ims"][t_step] = np.load(conf["in_npy_path"]+names[t_step]+".npy")
			deb.prints(patch["full_ims"].dtype)
			deb.prints(np.average(patch["full_ims"][t_step]))
			deb.prints(np.max(patch["full_ims"][t_step]))
			deb.prints(np.min(patch["full_ims"][t_step]))
			
			#deb.prints(patch["full_ims"][t_step].dtype)
			patch["full_label_ims"][t_step] = cv2.imread(str(conf["path"]/(self.dataSource.label_folder+"/"+label_names[t_step]+".tif")),0)
			print(conf["path"]/(self.dataSource.label_folder+"/"+label_names[t_step]+".tif"))
			deb.prints(conf["path"]/(self.dataSource.label_folder+"/"+label_names[t_step]+".tif"))
			deb.prints(np.unique(patch["full_label_ims"][t_step],return_counts=True))

		ic(patch["full_ims"].shape)
		ic(patch["full_label_ims"].shape)
		ic(patch["full_ims"].dtype)
		ic(patch["full_label_ims"].dtype)
		
		deb.prints(np.unique(patch['full_label_ims'],return_counts=True))
		return patch

	def getChannelsToMask(self):
		return self.dataSource.channelsToMask
class CampoVerde(Dataset):
	def __init__(self, seq_mode=None, seq_date=None, seq_len=12):
		name='cv'
		path="../cv_data/"
		class_n=13
		im_h=8492
		im_w=7995
		class_list = ['Background','Soybean','Maize','Cotton','Sorghum','Beans','NCC','Pasture','Eucaplyptus','Soil','Turfgrass','Cerrado']
		padded_dates = [-14, -13, -12, -11]
#		super().__init__(path,im_h,im_w,class_n,class_list,name)
		scaler_load=False
		scaler_name=name
		super().__init__(path,im_h,im_w,class_n,class_list,name,padded_dates,seq_mode,seq_date,scaler_load,scaler_name,seq_len)
	def addDataSource(self,dataSource):
		self.dataSource = dataSource
		if self.dataSource.name == 'SARSource':

			mode='var'
			mode='fixed'
			im_list_full = ['20151029_S1', '20151110_S1', '20151122_S1', '20151204_S1', 
				'20151216_S1', '20160121_S1', '20160214_S1', '20160309_S1', '20160321_S1', 
				'20160508_S1', '20160520_S1', '20160613_S1', '20160707_S1', '20160731_S1']
			if self.seq_mode=='var':
				self.im_list=im_list_full.copy()
			elif self.seq_mode=='fixed':
				# 12 len fixed. label -5
				if self.seq_date=='jun':
					date_id = 3
					self.im_list=im_list_full[-date_id-12+1:-date_id+1]
					assert len(self.im_list)==12
					assert self.im_list[-1]=='20160613_S1'
				elif self.seq_date=='may':
					date_id = 4
					self.im_list=im_list_full[-date_id-11+1:-date_id+1]
					assert len(self.im_list)==11
					assert self.im_list[-1]=='20160520_S1'

				elif self.seq_date=='mar':
					date_id = 6
					self.im_list=im_list_full[-date_id-9+1:-date_id+1]
					assert len(self.im_list)==9
					assert self.im_list[-1]=='20160321_S1'
				elif self.seq_date=='feb':
					date_id = 7
					self.im_list=im_list_full[-date_id-7+1:-date_id+1]
					assert len(self.im_list)==7
					assert self.im_list[-1]=='20160214_S1'
				elif self.seq_date=='jan':
					date_id = 8
					self.im_list=im_list_full[-date_id-6+1:-date_id+1]
					assert len(self.im_list)==6
					assert self.im_list[-1]=='20160121_S1'
				elif self.seq_date=='dec':
					date_id = 10
					self.im_list=im_list_full[-date_id-5+1:-date_id+1]
					ic(self.im_list)
					assert len(self.im_list)==5
					assert self.im_list[-1]=='20151216_S1'
				elif self.seq_date=='nov':
					date_id = 12
					self.im_list=im_list_full[-date_id-3+1:-date_id+1]
					assert len(self.im_list)==3
					assert self.im_list[-1]=='20151122_S1'
				elif self.seq_date=='oct':
					date_id = 14
					self.im_list=im_list_full[-date_id-1+1:-date_id+1]
					assert len(self.im_list)==1
					assert self.im_list[-1]=='20151029_S1'					
		elif self.dataSource.name == 'OpticalSource':
			self.im_list=[]
			self.label_list=self.im_list.copy()
		self.t_len=len(self.im_list)
		self.label_list=self.im_list.copy()
class LEM(Dataset):
	def __init__(self, seq_mode=None, seq_date=None, seq_len=12):
		name='lm'
		path="../lm_data/"
		class_n=15
		im_w=8658
		im_h=8484
		class_list = ['Background','Soybean','Maize','Cotton','Coffee','Beans','Sorghum','Millet','Eucalyptus','Pasture/Grass','Hay','Cerrado','Conversion Area','Soil','Not Identified']
		padded_dates = [-12, -11]
		scaler_load=False
		scaler_name=name
		super().__init__(path,im_h,im_w,class_n,class_list,name,padded_dates,seq_mode,seq_date,scaler_load,scaler_name,seq_len)

	def addDataSource(self,dataSource):
		deb.prints(dataSource.name)
		self.dataSource = dataSource
		if self.dataSource.name == 'SARSource':
			mode='fixed'
			
			im_list_full = ['20160927_S1','20161015_S1','20161120_S1','20161214_S1','20170119_S1','20170212_S1','20170308_S1',
					'20170413_S1','20170519_S1',
					'20170612_S1', '20170706_S1', '20170811_S1', '20170916_S1', '20171010_S1', '20171115_S1', 
					'20171209_S1', '20180114_S1', '20180219_S1', '20180315_S1', '20180420_S1', '20180514_S1',
					]
			if self.seq_mode=='var':
				self.im_list=im_list_full.copy()
			elif self.seq_mode=='fixed':
				# 12 len fixed. label -5
				if self.seq_date=='jan':
					self.im_list=im_list_full[-5-12+1:-5+1]
					assert len(self.im_list)==12
					assert self.im_list[-1]=='20180114_S1'
				elif self.seq_date=='feb':					
					self.im_list=im_list_full[-4-12+1:-4+1]
					assert len(self.im_list)==12
					assert self.im_list[-1]=='20180219_S1'
				elif self.seq_date=='mar':					
					self.im_list=im_list_full[-3-self.seq_len+1:-3+1]
					assert len(self.im_list)==self.seq_len
					assert self.im_list[-1]=='20180315_S1'
				elif self.seq_date=='apr':					
					self.im_list=im_list_full[-2-12+1:-2+1]
					assert len(self.im_list)==12
					assert self.im_list[-1]=='20180420_S1'

				elif self.seq_date=='may':					
					self.im_list=im_list_full[-1-12+1:]
					assert len(self.im_list)==12
					assert self.im_list[-1]=='20180514_S1'

				elif self.seq_date=='jun':					
					self.im_list=im_list_full[-12-12+1:-12+1]
					# assert len(self.im_list)==12
					assert self.im_list[-1]=='20170612_S1'

				elif self.seq_date=='jul':					
					self.im_list=im_list_full[-11-12+1:-11+1]
					# assert len(self.im_list)==12
					assert self.im_list[-1]=='20170706_S1'

				elif self.seq_date=='aug':					
					self.im_list=im_list_full[-10-12+1:-10+1]
					assert len(self.im_list)==12
					assert self.im_list[-1]=='20170811_S1'

				elif self.seq_date=='sep':					
					self.im_list=im_list_full[-9-12+1:-9+1]
					assert len(self.im_list)==12
					assert self.im_list[-1]=='20170916_S1'

				elif self.seq_date=='oct':					
					self.im_list=im_list_full[-8-12+1:-8+1]
					assert len(self.im_list)==12
					assert self.im_list[-1]=='20171010_S1'
				elif self.seq_date=='nov':					
					self.im_list=im_list_full[-7-12+1:-7+1]
					assert len(self.im_list)==12
					assert self.im_list[-1]=='20171115_S1'
				elif self.seq_date=='dec':
					self.im_list=['20170119_S1','20170212_S1','20170308_S1',
						'20170413_S1','20170519_S1',
						'20170612_S1', '20170706_S1', '20170811_S1', '20170916_S1', '20171010_S1', '20171115_S1', 
						'20171209_S1'
						]
			
			print("self.im_list",self.im_list)
			print(self.seq_mode, self.seq_date)
			self.label_list=self.im_list.copy()
		self.t_len=len(self.im_list)
		
		deb.prints(self.t_len)

class LEM2(Dataset):
	def __init__(self, seq_mode=None, seq_date=None, seq_len=12):
		name='l2'
		path="../l2_data/"
		class_n=15
		im_w=8658
		im_h=8484
		class_list = ['Background','Soybean','Maize','Cotton','Coffee','Beans','Sorghum','Millet','Eucalyptus','Pasture/Grass','Hay','Cerrado','Conversion Area','Soil','Not Identified']
		padded_dates = []
		scaler_load=True
		scaler_name='lm'
		super().__init__(path,im_h,im_w,class_n,class_list,name,padded_dates,seq_mode,seq_date,scaler_load,scaler_name,seq_len)

	def addDataSource(self,dataSource):
		deb.prints(dataSource.name)
		self.dataSource = dataSource
		if self.dataSource.name == 'SARSource':
			deb.prints(self.seq_mode)
			deb.prints(self.seq_date)

			mode='fixed'
			im_list_full = ['20181110_S1','20181216_S1','20190121_S1','20190214_S1','20190322_S1',
				'20190415_S1','20190521_S1','20190614_S1','20190720_S1','20190813_S1','20190918_S1',
				'20191012_S1','20191117_S1','20191223_S1','20200116_S1','20200221_S1','20200316_S1',
				'20200421_S1','20200515_S1','20200620_S1','20200714_S1','20200819_S1','20200912_S1']

			if self.seq_mode=='var_label':
				self.im_list=im_list_full.copy()
			elif self.seq_mode=='fixed':

				if self.seq_date=='jan':
					id_=-9
					self.im_list=im_list_full[id_-12+1:id_+1]
					assert len(self.im_list)==12
					assert self.im_list[-1]=='20200116_S1'

				elif self.seq_date=='feb':					
					id_=-8
					self.im_list=im_list_full[id_-12+1:id_+1]
					assert len(self.im_list)==12
					assert self.im_list[-1]=='20200221_S1'
				elif self.seq_date=='mar':					
					id_=-7
					self.im_list=im_list_full[id_-12+1:id_+1]
					assert len(self.im_list)==12
					assert self.im_list[-1]=='20200316_S1'
				elif self.seq_date=='apr':					
					id_=-6
					self.im_list=im_list_full[id_-12+1:id_+1]
					assert len(self.im_list)==12
					assert self.im_list[-1]=='20200421_S1'
				elif self.seq_date=='may':					
					id_=-5
					self.im_list=im_list_full[id_-12+1:id_+1]
					assert len(self.im_list)==12
					assert self.im_list[-1]=='20200515_S1'
				elif self.seq_date=='jun':					
					id_=-4
					self.im_list=im_list_full[id_-12+1:id_+1]
					assert len(self.im_list)==12
					assert self.im_list[-1]=='20200620_S1'
				elif self.seq_date=='jul':					
					id_=-3
					self.im_list=im_list_full[id_-12+1:id_+1]
					assert len(self.im_list)==12
					assert self.im_list[-1]=='20200714_S1'
				elif self.seq_date=='aug':					
					id_=-2
					self.im_list=im_list_full[id_-12+1:id_+1]
					assert len(self.im_list)==12
					assert self.im_list[-1]=='20200819_S1'
				elif self.seq_date=='sep':					
					id_=-1
					self.im_list=im_list_full[id_-12+1:]
					assert len(self.im_list)==12
					assert self.im_list[-1]=='20200912_S1'
				elif self.seq_date=='oct':					
					id_=-12
					self.im_list=im_list_full[id_-12+1:id_+1]
					assert len(self.im_list)==12
					assert self.im_list[-1]=='20191012_S1'
				elif self.seq_date=='nov':					
					id_=-11
					self.im_list=im_list_full[id_-12+1:id_+1]
					assert len(self.im_list)==12
					assert self.im_list[-1]=='20191117_S1'
				elif self.seq_date=='dec':					
					id_=-10
					self.im_list=im_list_full[id_-12+1:id_+1]
					assert len(self.im_list)==12
					assert self.im_list[-1]=='20191223_S1'


			self.label_list=self.im_list.copy()

		self.t_len=len(self.im_list)
		
		deb.prints(self.t_len)



