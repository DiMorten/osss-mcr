from dataSource import DataSource, SARSource, Dataset, LEM, LEM2, CampoVerde
from model_input_mode import MIMFixed, MIMVarLabel, MIMVarSeqLabel, MIMVarLabel_PaddedSeq, MIMFixedLabelAllLabels, MIMFixed_PaddedSeq
from parameters.params_train import ParamsTrain
from parameters.params_mosaic import ParamsReconstruct
from icecream import ic
from monitor import Monitor, MonitorNPY, MonitorGenerator, MonitorNPYAndGenerator
from model import ModelLoadGeneratorWithCoords
from dataset import Dataset, DatasetWithCoords

from patch_extractor import PatchExtractor
from train_and_evaluate import TrainTest

if __name__ == '__main__':

	paramsTrainCustom = {
		'getFullIms': False, # only True if first time
		'coordsExtract': False, # only True if first time
		'train': False,
		'openSetMethod': 'SoftmaxThresholding', # OpenPCS, SoftmaxThresholding, OpenPCS++
		'openSetLoadModel': False,
		'selectMainClasses': True,
		'dataset': 'lm',
		'seq_date': 'mar'		
	}

	paramsTrain = ParamsTrain('parameters/', **paramsTrainCustom)

	paramsTrain.dataSource = SARSource()

	trainTest = TrainTest(paramsTrain)

	trainTest.main()



