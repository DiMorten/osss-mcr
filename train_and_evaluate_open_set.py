from src.dataSource import DataSource, SARSource, Dataset, LEM, LEM2, CampoVerde
from src.model_input_mode import MIMFixed, MIMVarLabel, MIMVarSeqLabel, MIMVarLabel_PaddedSeq, MIMFixedLabelAllLabels, MIMFixed_PaddedSeq
from parameters.params_train import ParamsTrain
from parameters.params_mosaic import ParamsReconstruct
from icecream import ic
from src.monitor import Monitor, MonitorNPY, MonitorGenerator, MonitorNPYAndGenerator
from src.modelManager import ModelManagerCropRecognition
from src.dataset import Dataset, DatasetWithCoords

from src.patch_extractor import PatchExtractor
from train_and_evaluate import TrainTest
from src.modelArchitecture import UUnetConvLSTM, UnetSelfAttention, UUnetConvLSTMDropout, UUnetConvLSTMEvidential

if __name__ == '__main__':


	paramsTrainCustom = {
		'getFullIms': False, # only True if first time
		'coordsExtract': False, # only True if first time
		'train': False,
		'openSetMethod': 'OpenPCS++', # OpenPCS, SoftmaxThresholding, OpenPCS++
		'confidenceScaling': False,
		'applyThreshold': False,
		'openSetLoadModel': True,
		'selectMainClasses': True,
		'dataset': 'lm',
		'seq_date': 'mar',	# jun, mar	
#		'id': 'evidential4',
#		'model_type': UUnetConvLSTMEvidential
		'id': 'focal_lr1em3', # temperaturescaling, focal, focal_lr1em3
		'model_type': UUnetConvLSTM
	}


	if paramsTrainCustom['confidenceScaling'] == True:
		paramsTrainCustom['val_set'] = True

	paramsTrain = ParamsTrain('parameters/', **paramsTrainCustom)

	paramsTrain.dataSource = SARSource()

	trainTest = TrainTest(paramsTrain)

	trainTest.main()



