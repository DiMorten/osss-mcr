from src.dataSource import DataSource, SARSource, Dataset, LEM, LEM2, CampoVerde
from src.model_input_mode import MIMFixed, MIMVarLabel, MIMVarSeqLabel, MIMVarLabel_PaddedSeq, MIMFixedLabelAllLabels, MIMFixed_PaddedSeq
from parameters.params_train import ParamsTrain
from parameters.params_mosaic import ParamsReconstruct
from icecream import ic
from src.monitor import Monitor, MonitorNPY, MonitorGenerator, MonitorNPYAndGenerator
from src.model import ModelCropRecognition
from src.dataset import Dataset, DatasetWithCoords

from src.patch_extractor import PatchExtractor
from train_and_evaluate import TrainTest, TrainTestDropout

from src.modelArchitecture import UUnetConvLSTM, UnetSelfAttention, UUnetConvLSTMDropout, UUnetConvLSTMDropout2, UUnetConvLSTMEvidential


if __name__ == '__main__':

	paramsTrainCustom = {
		'getFullIms': False, # only True if first time
		'coordsExtract': False, # only True if first time
		'train': False,
		'openSetMethod': None, # OpenPCS, SoftmaxThresholding, OpenPCS++
		'dropoutInference': True,
		'openSetLoadModel': False,
		'selectMainClasses': True,
		'dataset': 'lm',
		'seq_date': 'mar',	# jun, mar	
		'id': 'dropout1',
		'model_type': UUnetConvLSTMDropout2,
	}

	paramsTrain = ParamsTrain('parameters/', **paramsTrainCustom)

	paramsTrain.dataSource = SARSource()

	trainTest = TrainTestDropout(paramsTrain)

	trainTest.main()



