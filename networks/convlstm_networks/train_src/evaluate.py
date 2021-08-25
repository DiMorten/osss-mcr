from dataSource import DataSource, SARSource, OpticalSource, Dataset, LEM, LEM2, CampoVerde, OpticalSourceWithClouds, Humidity
from model_input_mode import MIMFixed, MIMVarLabel, MIMVarSeqLabel, MIMVarLabel_PaddedSeq, MIMFixedLabelAllLabels, MIMFixed_PaddedSeq
from parameters.parameters_reader import ParamsTrain
from parameters.params_reconstruct import ParamsReconstruct
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
		'openSetMethod': 'OpenPCS++',
		'openSetLoadModel': True
	}

	paramsTrain = ParamsTrain('parameters/', **paramsTrainCustom)
	
	dataset = paramsTrain.dataset

	paramsTrain.dataSource = SARSource()

	trainTest = TrainTest(paramsTrain)
	
	trainTest.setData()

	trainTest.preprocess(paramsTrain.model_name_id) # move into if

	trainTest.setModel()
	
	trainTest.modelLoad(paramsTrain.model_name_id)

	if paramsTrain.openSetMethod != None:
		trainTest.setPostProcessing()
		trainTest.fitPostProcessing()


	paramsMosaic = ParamsReconstruct(paramsTrain)

	trainTest.mosaicCreate(paramsMosaic)
	trainTest.evaluate()



