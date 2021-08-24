from icecream import ic
import numpy as np
from pathlib import Path
class ParamsReconstruct():
	def __init__(self, paramsTrain):

		self.spatial_results_path = Path('results/spatial_results')
		self.paramsTrain = paramsTrain

		self.mosaic_flag = True

		self.open_set_mode = False if self.paramsTrain.openSetMethod == None else True

		self.prediction_mask = False
#		self.conditionType = 'test'
		self.conditionType = 'all'

		self.metrics_flag = False

		self.prediction_type = 'model'
		self.save_input_im = False

		self.croppedFlag = False



		self.threshold_idx = 4
		self.overlap = self.paramsTrain.test_overlap_percentage # 0.5

		if self.overlap > 0:
			self.add_padding_flag = True
		else:
			self.add_padding_flag = False

		self.overlap_mode = 'replace' # average, replace
#		self.overlap_mode = 'average' # average, replace
#        self.overlap_mode = 'average_score' # average, replace

		if self.croppedFlag == True:
			self.add_padding_flag = False
			self.overlap = 0

		ic(self.overlap, self.threshold_idx)




		self.data_path='../../' / self.paramsTrain.path
		self.setModelPath('../model/') 

		self.prediction_dtype = np.float16

		self.label_entire_save = False
	def setModelPath(self, model_path):
				
		self.model_path=model_path
		
		if self.paramsTrain.model_type == 'UUnet4ConvLSTM':
			self.paramsTrain.model_type_specific = self.paramsTrain.model_type
			self.paramsTrain.model_type = 'unet'
		if self.paramsTrain.dataset=='lm':

			self.model_path+='lm/'
			ic(self.paramsTrain.model_type, self.paramsTrain.seq_date)
			if self.paramsTrain.model_type=='densenet':
				self.predictions_path=self.model_path+'prediction_DenseNetTimeDistributed_128x2_batch16_full.npy'
			elif self.paramsTrain.model_type=='biconvlstm':
				self.predictions_path=self.model_path+'prediction_ConvLSTM_seq2seq_bi_batch16_full.npy'
			elif self.paramsTrain.model_type=='convlstm':
				self.predictions_path=self.model_path+'prediction_ConvLSTM_seq2seq_batch16_full.npy'
			elif self.paramsTrain.model_type=='unet':
				self.predictions_path=self.model_path+'prediction_BUnet4ConvLSTM_repeating1.npy'
				#self.predictions_path=self.model_path+'prediction_BUnet4ConvLSTM_repeating2.npy'
				#self.predictions_path=self.model_path+'prediction_BUnet4ConvLSTM_repeating4.npy'
				self.predictions_path = self.model_path+'model_best_UUnet4ConvLSTM_doty_fixed_label_fixed_'+self.paramsTrain.seq_date+'_700perclass.h5'			
				self.predictions_path = self.model_path+'model_best_UUnet4ConvLSTM_fixed_label_fixed_'+self.paramsTrain.seq_date+'_loco8_lm_testlm_fewknownclasses.h5'	
				if self.paramsTrain.seq_date == 'mar':
					self.predictions_path = self.model_path+'model_best_UUnet4ConvLSTM_fixed_label_fixed_'+self.paramsTrain.seq_date+'_loco8_lm_testlm_fewknownclasses.h5'	
					self.predictions_path = self.model_path+'model_lm_mar_nomask_good.h5'	
					#self.predictions_path = self.model_path+'model_best_UUnet4ConvLSTM_mar_lm_fixed_fewknownclasses_groupclasses_newdataaugmentation_coords.h5'
				elif self.paramsTrain.seq_date == 'jun':
					self.predictions_path = self.model_path+'model_best_UUnet4ConvLSTM_fixed_label_fixed_jun_lm_fewknownclasses2.h5'	
					self.predictions_path = self.model_path+'model_best_UUnet4ConvLSTM_jun_lm_.h5'	
#                    self.predictions_path = self.model_path+'model_best_UUnet4ConvLSTM_jun_cv_criteria_0_92.h5'	

					


			elif self.paramsTrain.model_type=='atrous':
				self.predictions_path=self.model_path+'prediction_BAtrousConvLSTM_2convins5.npy'
			elif self.paramsTrain.model_type=='atrousgap':
				self.predictions_path=self.model_path+'prediction_BAtrousGAPConvLSTM_raulapproved.npy'
				#self.predictions_path=self.model_path+'prediction_BAtrousGAPConvLSTM_repeating3.npy'
				#self.predictions_path=self.model_path+'prediction_BAtrousGAPConvLSTM_repeating4.npy'
				
			self.predictions_path = self.model_path+'model_best_' + str(self.paramsTrain.model_type) + '_' + \
				self.paramsTrain.seq_date + '_' + self.paramsTrain.dataset + '_' + \
				self.paramsTrain.model_name + '.h5'
#			self.predictions_path = self.model_path+'model_best_' + self.paramsTrain.model_type_specific + '_' + \
#				self.paramsTrain.seq_date + '_' + self.paramsTrain.dataset + '_' + \
#				self.paramsTrain.model_name + '.h5'

			ic(self.predictions_path)

			self.mask_path=self.data_path / 'TrainTestMask.tif'
			self.location_path=self.data_path / 'locations/'
			self.folder_load_path=self.data_path / 'train_test/test/labels/'

			self.custom_colormap = np.array([[255,146,36],
							[255,255,0],
							[164,164,164],
							[255,62,62],
							[0,0,0],
							[172,89,255],
							[0,166,83],
							[40,255,40],
							[187,122,83],
							[217,64,238],
							[0,113,225],
							[128,0,0],
							[114,114,56],
							[53,255,255]])
		elif self.paramsTrain.dataset=='cv':

			self.model_path+='cv/'
			if self.paramsTrain.model_type=='densenet':
				self.predictions_path=self.model_path+'prediction_DenseNetTimeDistributed_128x2_batch16_full.npy'
			elif self.paramsTrain.model_type=='biconvlstm':
				self.predictions_path=self.model_path+'prediction_ConvLSTM_seq2seq_bi_batch16_full.npy'
			elif self.paramsTrain.model_type=='convlstm':
				self.predictions_path=self.model_path+'prediction_ConvLSTM_seq2seq_batch16_full.npy'		
			elif self.paramsTrain.model_type=='unet':
				#self.predictions_path=self.model_path+'prediction_BUnet4ConvLSTM_repeating2.npy'
				self.predictions_path=self.model_path+'model_best_BUnet4ConvLSTM_int16.h5'
				if self.paramsTrain.seq_date == 'jun':
					self.predictions_path = self.model_path+'model_best_UUnet4ConvLSTM_jun.h5'
					self.predictions_path = self.model_path+'model_best_UUnet4ConvLSTM_jun_cv_criteria_0_92.h5'
				if self.paramsTrain.seq_date == 'may':
					self.predictions_path = self.model_path+'model_cv_may_3classes_nomask.h5'
			elif self.paramsTrain.model_type=='atrous':
				self.predictions_path=self.model_path+'prediction_BAtrousConvLSTM_repeating2.npy'			
			elif self.paramsTrain.model_type=='atrousgap':
				#self.predictions_path=self.model_path+'prediction_BAtrousGAPConvLSTM_raulapproved.npy'			
				#self.predictions_path=self.model_path+'prediction_BAtrousGAPConvLSTM_repeating4.npy'			
				self.predictions_path=self.model_path+'prediction_BAtrousGAPConvLSTM_repeating6.npy'			
			elif self.paramsTrain.model_type=='unetend':
				self.predictions_path=self.model_path+'prediction_unet_convlstm_temouri2.npy'			
			elif self.paramsTrain.model_type=='allinputs':
				self.predictions_path=self.model_path+'prediction_bconvlstm_wholeinput.npy'			

			self.mask_path=self.data_path / 'TrainTestMask.tif'
			self.location_path=self.data_path / 'locations/'

			self.folder_load_path=self.data_path / 'train_test/test/labels/'

			self.custom_colormap = np.array([[255, 146, 36],
						[255, 255, 0],
						[164, 164, 164],
						[255, 62, 62],
						[0, 0, 0],
						[172, 89, 255],
						[0, 166, 83],
						[40, 255, 40],
						[187, 122, 83],
						[217, 64, 238],
						[45, 150, 255]])
		elif self.paramsTrain.dataset=='l2':
			self.model_path+='l2/'	
			if self.paramsTrain.model_type=='unet':
				self.predictions_path=self.model_path+'prediction_BUnet4ConvLSTM_repeating1.npy'
				#self.predictions_path=self.model_path+'prediction_BUnet4ConvLSTM_repeating2.npy'
				#self.predictions_path=self.model_path+'prediction_BUnet4ConvLSTM_repeating4.npy'
				self.predictions_path = self.model_path+'model_best_UUnet4ConvLSTM_doty_fixed_label_dec.h5'
				self.predictions_path = self.model_path+'model_best_UUnet4ConvLSTM_doty_fixed_label_fixed_'+self.paramsTrain.seq_date+'.h5'
				self.predictions_path = self.model_path+'model_best_UUnet4ConvLSTM_doty_fixed_label_fixed_'+self.paramsTrain.seq_date+'_700perclass.h5'
		#		self.predictions_path = self.model_path+'model_best_UUnet4ConvLSTM_doty_fixed_label_dec_good_slvc05.h5'
			self.mask_path=self.data_path / 'TrainTestMask.tif'
			self.location_path=self.data_path / 'locations/'
			self.folder_load_path=self.data_path / 'train_test/test/labels/'

			self.custom_colormap = np.array([[255,146,36],
							[255,255,0],
							[164,164,164],
							[255,62,62],
							[0,0,0],
							[172,89,255],
							[0,166,83],
							[40,255,40],
							[187,122,83],
							[217,64,238],
							[0,113,225],
							[128,0,0],
							[114,114,56],
							[53,255,255]])



