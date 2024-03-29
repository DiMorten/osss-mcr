import os
import json
from icecream import ic
import pdb
from pathlib import Path
import sys
sys.path.append('src/')
from modelArchitecture import UUnetConvLSTM, UnetSelfAttention, UUnetConvLSTMDropout, UUnetConvLSTMEvidential
from model_input_mode import MIMFixed, MIMVarLabel, MIMVarSeqLabel, MIMVarLabel_PaddedSeq, MIMFixedLabelAllLabels, MIMFixed_PaddedSeq
import deb
from icecream import ic

class Params():
	"""Class that loads hyperparameters from a json file.

	Example:
	```
	params = Params(json_path)
	print(params.learning_rate)
	params.learning_rate = 0.5  # change the value of learning_rate in params
	```
	"""

	def __init__(self, json_path):

		assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
		with open(json_path) as f:
			params = json.load(f)
			self.__dict__.update(params)

	def save(self, json_path):
		with open(json_path, 'w') as f:
			json.dump(self.__dict__, f, indent=4)
			
	def update(self, json_path):
		"""Loads parameters from json file"""
		with open(json_path) as f:
			params = json.load(f)
			self.__dict__.update(params)

	@property
	def dict(self):
		"""Gives dict-like access to Params instance by `params.dict['learning_rate']"""
		return self.__dict__

class ParamsTrain(Params):
	def __init__(self, folder_path, **kwargs):

		# ============= PATCH EXTRACTION ============== #

		self.getFullIms = False if ('getFullIms' not in kwargs.keys()) else kwargs['getFullIms']
		self.coordsExtract = False if ('coordsExtract' not in kwargs.keys()) else kwargs['coordsExtract']
		self.train = False if ('train' not in kwargs.keys()) else kwargs['train']
		self.openSetLoadModel = False if ('openSetLoadModel' not in kwargs.keys()) else kwargs['openSetLoadModel']
		self.model_type = UUnetConvLSTM if ('model_type' not in kwargs.keys()) else kwargs['model_type']

		self.dropoutInference = False if ('dropoutInference' not in kwargs.keys()) else kwargs['dropoutInference']
		self.evidentialDL = False if ('evidentialDL' not in kwargs.keys()) else kwargs['evidentialDL']
		ic(self.evidentialDL)
#        self.model_load = True

		self.train_overlap_percentage = 0

		if self.train_overlap_percentage>0:
			self.trainGeneratorRandom = True
		else:
			self.trainGeneratorRandom = False

		self.patch_len = 32

		self.test_overlap_percentage = 0

		# options: None, OpenPCS, OpenPCS++
		self.openSetMethod = 'OpenPCS++' if ('openSetMethod' not in kwargs.keys()) else kwargs['openSetMethod']	
		self.selectMainClasses = True if ('selectMainClasses' not in kwargs.keys()) else kwargs['selectMainClasses']
		self.confidenceScaling = False if ('confidenceScaling' not in kwargs.keys()) else kwargs['confidenceScaling']	

		# ============== SCRIPT MODE: CLOSED SET, OPEN SET... ================= #

		if self.openSetMethod == None:
			self.openMode = 'NoMode'
#		elif self.openSetMethod == 'OpenPCS' or self.openSetMethod == 'OpenPCS++':
		else:
			if self.train == False:
				self.openMode = 'SaveNonaugmentedTrainPatches'
			else:
				self.openMode = 'OpenSet'

		
		
		if self.openMode == 'OpenSet':
			json_path = folder_path+'parameters_openset.json'
		elif self.openMode == 'ClosedSetGroupClasses':
			json_path = folder_path+'parameters_closedset_groupclasses.json'
		elif self.openMode == 'SaveNonaugmentedTrainPatches':
			json_path = folder_path+'save_nonaugmented_train_patches.json'
		elif self.openMode == 'NoMode':
			json_path = folder_path+'no_mode.json'

		# DATASET AND DATA
		self.dataset = 'lm'  if ('dataset' not in kwargs.keys()) else kwargs['dataset'] # LM: jun, mar, dec. CV: jun, sep

		self.seq_date = 'mar'  if ('seq_date' not in kwargs.keys()) else kwargs['seq_date']
		
#		self.model_name = 'dummy'
#		self.id = 'focal'
		self.id = 'dummy' if ('id' not in kwargs.keys()) else kwargs['id']

		ic(self.id)
		self.model_name = self.id

		self.learning_rate = 0.0001
		
		

		ic(self.seq_date)
#        pdb.set_trace()
		self.known_classes = []
		if self.dataset == 'lm':
			if self.seq_date == 'mar':
				self.known_classes = [0, 1, 10, 12]
#                
#                self.samples_per_class = 700

			elif self.seq_date == 'jun':
#                self.known_classes = [1, 6, 10, 12]
				self.known_classes = [1, 5, 6, 10, 12]

			if self.openMode == 'NoMode':
				self.known_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
		elif self.dataset == 'cv':
			if self.seq_date == 'may':
				self.known_classes = [1, 2, 6]
			elif self.seq_date == 'jun':
				#self.known_classes = [1, 2, 8]
				self.known_classes = [1, 2, 6, 8]
		print("self.known_classes", self.known_classes)
		self.known_classes_percentage = 0.92

		# default main params

		self.debug = 1
		self.epochs = 8000
		self.patience = 10
		self.eval_mode = "metrics" # legacy
		self.im_store = True # legacy
		self.exp_id = "default" # legacy.
		self.save_patches_only = False # legacy bc of coords

		self.time_measure = False


		# General params (common to open set and closed set and group classes)

		if self.dataset == 'lm':
			#self.path = "../../../dataset/dataset/lm_data/"
			self.class_n = 15
			self.t_len = 19
		elif self.dataset == 'cv':
			#self.path = "../../../dataset/dataset/cv_data/"
			self.class_n = 12
			self.t_len = 14
		elif self.dataset == 'l2':
			#self.path = "../../../dataset/dataset/l2_data/"
			self.class_n = 15

		self.channel_n = 2

		self.stop_epoch = 400

		
		#self.stride = self.patch_len

		self.stride = int(self.patch_len - self.patch_len * self.train_overlap_percentage)
		ic(self.stride)


		self.patch_step_train = self.stride
		self.patch_step_test = self.patch_len
		self.batch_size_train = 16
		self.batch_size_test = 16  #unused
		self.t_len = 12 # variable? depends on dataset?
		self.model_t_len = 12
		# usually editable params
		# self.dropout_mode = True
		
		if self.dropoutInference == False:
#			model_type = UUnetConvLSTM # Options: UUnetConvLSTM, UnetSelfAttention
#			model_type = UUnetConvLSTMEvidential

			self.model_type = self.model_type(self.model_t_len, self.patch_len, self.channel_n)
		else:
			self.dropout = 0.2
			#self.model_type = UUnetConvLSTMDropout

			self.model_type = self.model_type(self.model_t_len, self.patch_len, self.channel_n,
				self.dropout)

#        self.seq_mode = "fixed"
		#self.seq_date = "mar"
		
		self.loco_class = 8 # legacy. delete


		self.path = Path("E:/Jorge/dataset_crop_recognition") / (self.dataset + "_data")
		# self.path = Path("dataset/") / (self.dataset + "_data")
#		self.path = Path(
#			'D:/Jorg/phd/convlstm_crop/classification_n_to_1/FCN_ConvLSTM_Crop_Recognition_Generalized/dataset/dataset') / (self.dataset + "_data")

		print(os.listdir(folder_path))
		super().__init__(json_path)        


		if self.openMode == 'NoMode' and self.selectMainClasses == True:
			self.select_kept_classes_flag = True

		if self.seq_mode == 'var_label':
			#self.mim = MIMVarLabel()
			self.mim = MIMVarLabel_PaddedSeq()
		elif self.seq_mode == 'var':
			self.mim = MIMVarSeqLabel()
		elif self.seq_mode == 'fixed_label_len':
			self.mim = MIMVarLabel()
			self.mim =MIMFixedLabelAllLabels()
		else:
			#self.mim = MIMFixed()
			self.mim = MIMFixed_PaddedSeq()

		deb.prints(self.seq_mode)
		deb.prints(self.mim)

#		self.model_path = Path('../results/convlstm_results/model/lm/')
		self.model_path = Path('results/model/' + self.dataset + '/')

		self.modelNameSpecify = True

		if self.modelNameSpecify == True:
			assert isinstance(str(self.model_type), str)
#			self.model_name_id = self.model_path / ('model_best_' + str(self.model_type) + '_' + \
#					self.seq_date + '_' + self.dataset + '_' + \
#					self.model_name + '_' + str(self.openSetMethod) + '.h5')
			self.model_name_id = self.model_path / ('model_best_' + str(self.model_type) + '_' + \
					self.seq_date + '_' + self.dataset + '_' + \
					self.model_name + '.h5')

		else:
			self.model_name_id = self.model_path / 'model_best_fit.h5'
			self.model_name_id = self.model_path / 'model_best_UUnetConvLSTM_mar_lm_dummy.h5'

			
#			self.model_name_id = self.model_path / 'model_lm_mar_nomask_good.h5'
##			self.model_name_id = self.model_path / 'model_best_UUnetConvLSTM_mar_lm_nomode.h5'

class ParamsAnalysis(Params):
	def __init__(self, folder_path):

		json_path = folder_path+'parameters_analysis_closedset.json'

		print(os.listdir(folder_path))
		super().__init__(json_path)