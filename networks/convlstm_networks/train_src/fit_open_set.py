from dataSource import DataSource, SARSource, OpticalSource, Dataset, LEM, LEM2, CampoVerde, OpticalSourceWithClouds, Humidity
from model_input_mode import MIMFixed, MIMVarLabel, MIMVarSeqLabel, MIMVarLabel_PaddedSeq, MIMFixedLabelAllLabels, MIMFixed_PaddedSeq
from parameters.parameters_reader import ParamsTrain
from parameters.params_reconstruct import ParamsReconstruct
from icecream import ic
from monitor import Monitor, MonitorNPY, MonitorGenerator, MonitorNPYAndGenerator
from model import ModelLoadGeneratorWithCoords
from dataset import Dataset, DatasetWithCoords

from patch_extractor import PatchExtractor
from main import TrainTest

if __name__ == '__main__':

	openSetMethod = 'OpenPCS++'

	paramsTrain = ParamsTrain('parameters/', openSetMethod = openSetMethod)
	
	dataset = paramsTrain.dataset

	paramsTrain.dataSource = SARSource()

	trainTest = TrainTest(paramsTrain)
	
	trainTest.setData()

	trainTest.preprocess(paramsTrain.model_name_id) # move into if

	trainTest.setModel()

	trainTest.modelLoad(paramsTrain.model_name_id)

	if openSetMethod != None:
		trainTest.fitOpenSet() 

	paramsMosaic = ParamsReconstruct(paramsTrain)
	trainTest.evaluate(paramsMosaic)



