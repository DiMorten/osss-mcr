#from natsort import natsorted, ns
import numpy as np
import pathlib
import deb
import glob

import re
import pdb
from pathlib import Path
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    return sorted(l, key=alphanum_key)


class PatchesStorage():
	def __init__(self):
		pass

class PatchesStorageAllSamples(PatchesStorage):
	def __init__(self,path, seq_mode, seq_date):
		
		self.path_patches = path + 'patches_bckndfixed/'
		self.path={}
		self.path['train_bckndfixed']=self.path_patches+'train/'
		self.path['val_bckndfixed']=self.path_patches+'val/'
		self.path['test_bckndfixed']=self.path_patches+'test/'
		self.path['test_loco']=self.path_patches+'test_loco/'
		self.path['train_loco'] = self.path_patches+'train_loco/'

		self.seq_mode = seq_mode
		self.seq_date = seq_date
		print("Path, ",self.path)
		#pdb.set_trace()
	def store(self,data_patches):
		self.storeSplit(data_patches['train'],'train_bckndfixed')
		self.storeSplit(data_patches['test'],'test_bckndfixed')
		#self.storeSplit(data_patches['val'],'val_bckndfixed')

	def storeSplit(self, patches, split='train_bckndfixed'):
		pathlib.Path(self.path[split]).mkdir(parents=True, exist_ok=True) 
		print("Storing in ",self.path[split])
		np.save(self.path[split]+'patches_in_'+self.seq_mode+'_'+self.seq_date+'.npy', patches['in']) #to-do: add polymorphism for other types of input 
		
		#pathlib.Path(self.path[split]['label']).mkdir(parents=True, exist_ok=True) 
		np.save(self.path[split]+'patches_label_'+self.seq_mode+'_'+self.seq_date+'.npy', patches['label']) #to-do: add polymorphism for other types of input 
		#pdb.set_trace()
	def load(self):
		data_patches={}
		data_patches['val']=self.loadSplit('val_bckndfixed')
		data_patches['train']=self.loadSplit('train_bckndfixed')
		data_patches['test']=self.loadSplit('test_bckndfixed')
		return data_patches

	def loadSplit(self, split='train'):
		out={}
		out['in']=np.load(self.path[split]+'patches_in.npy',mmap_mode='r')
		out['label']=np.load(self.path[split]+'patches_label.npy')
		return out

class PatchesStorageAllSamplesOpenSet(PatchesStorageAllSamples):
	def storeLabel(self, patches, split='test_loco'):
		pathlib.Path(self.path[split]).mkdir(parents=True, exist_ok=True) 
		print("Storing in ",self.path[split])
		
		#pathlib.Path(self.path[split]['label']).mkdir(parents=True, exist_ok=True) 
		np.save(self.path[split]+'patches_label_'+self.seq_mode+'_'+self.seq_date+'.npy', patches) #to-do: add polymorphism for other types of input 

	def store(self,data_patches):
		self.storeSplit(data_patches['train'],'train_bckndfixed')
		self.storeSplit(data_patches['test'],'test_bckndfixed')
		#self.storeSplit(data_patches['val'],'val_bckndfixed')
		self.storeLabel(data_patches['test']['label_with_loco_class'],'test_loco')
		self.storeLabel(data_patches['train']['label_with_loco_class'],'train_loco')



class PatchesStorageEachSample(PatchesStorage):
	def __init__(self,path):
		
		self.path_patches = path + 'patches_eachsample/' 
		self.path_im={}
		self.path_im['train']=self.path_patches+'im/'+'train/'
		self.path_im['val']=self.path_patches+'im/'+'val/'
		self.path_im['test']=self.path_patches+'im/'+'test/'
		self.path_label={}
		self.path_label['train']=self.path_patches+'label/'+'train/'
		self.path_label['val']=self.path_patches+'label/'+'val/'
		self.path_label['test']=self.path_patches+'label/'+'test/'

	def store(self,data_patches):
		self.storeSplit(data_patches['train'],'train')
		self.storeSplit(data_patches['test'],'test')
		self.storeSplit(data_patches['val'],'val')

	def storeSplit(self, patches, split='train'):
		pathlib.Path(self.path_im[split]).mkdir(parents=True, exist_ok=True) 
		pathlib.Path(self.path_label[split]).mkdir(parents=True, exist_ok=True) 

		for idx in range(patches['in'].shape[0]):
			np.save(self.path_im[split]+'patches_in'+str(idx).zfill(5)+'.npy', patches['in'][idx]) #to-do: add polymorphism for other types of input 
			np.save(self.path_label[split]+'patches_label'+str(idx).zfill(5)+'.npy', patches['label'][idx]) #to-do: add polymorphism for other types of input 
		pdb.set_trace()
		#pathlib.Path(self.path[split]['label']).mkdir(parents=True, exist_ok=True) 
		
	def load(self):
		# use self.folder_load from main.py
		data_patches={}
		data_patches['val']=self.loadSplit('val')
		data_patches['train']=self.loadSplit('train')
		data_patches['test']=self.loadSplit('test')
		return data_patches

	def loadSplit(self, split='train'):
		out={}
		out['in'],_=self.folder_load(self.path_im[split])
		out['label'],_=self.folder_load(self.path_label[split])
		return out
		
	def folder_load(self,folder_path): #move to patches_handler
		paths=glob.glob(folder_path+'*.npy')
		#deb.prints(paths)
		# sort in human order
		paths=natural_sort(paths)
		#deb.prints(paths)
		files=[]
		deb.prints(len(paths))
		for path in paths:
			#print(path)
			files.append(np.load(path))
		return np.asarray(files),paths
	def folder_load_partition(self,folder_path): #move to patches_handler
		paths=glob.glob(folder_path+'*.npy')
		#deb.prints(paths)
		# sort in human order
		paths=natural_sort(paths)
		return paths
	def loadSplitPartition(self, split='train'):
		partition={}
		partition['in']=self.folder_load_partition(self.path_im[split])
		partition['label']=self.folder_load_partition(self.path_label[split])
		return partition		
	def loadPartition(self):
		partition={}
		partition['val']=self.loadSplitPartition('val')
		partition['train']=self.loadSplitPartition('train')
		partition['test']=self.loadSplitPartition('test')
		return partition		