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
from icecream import ic
import pdb
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time
import matplotlib.pyplot as plt
import cv2
if __name__ == '__main__':

	paramsTrainCustom = {
		'getFullIms': False, # only True if first time
		'coordsExtract': False, # only True if first time
		'train': False,
		'openSetMethod': 'OpenPCS++', # OpenPCS, SoftmaxThresholding, OpenPCS++
		'openSetLoadModel': False,
		'selectMainClasses': True,
		'dataset': 'lm',
		'seq_date': 'mar',	# jun, mar	
#		'id': 'evidential4',
#		'model_type': UUnetConvLSTMEvidential
		'id': 'focal',
		'model_type': UUnetConvLSTM
	}

	paramsTrain = ParamsTrain('parameters/', **paramsTrainCustom)

	paramsTrain.dataSource = SARSource()

	trainTest = TrainTest(paramsTrain)


	patchExtractor = PatchExtractor(paramsTrain, trainTest.ds)	
	if paramsTrain.getFullIms == True:
		patchExtractor.getFullIms()	
	else:
		patchExtractor.fullImsLoad()

	if paramsTrain.coordsExtract == True:
		patchExtractor.extract()
	del patchExtractor
	
	trainTest.setData() 
	trainTest.data.loadMask()

	ic(trainTest.data.full_ims_test.shape)
	ic(trainTest.data.full_label_test.shape)
	ic(trainTest.data.mask.shape)



	label_list = ['20170612_S1',
		'20170706_S1',
		'20170811_S1',
		'20170916_S1',
		'20171010_S1',
		'20171115_S1',
		'20171209_S1',
		'20180114_S1',
		'20180219_S1',
		'20180315_S1',
		'20180420_S1',
		'20180514_S1',
		'20180619_S1']

	class_names = {1: 'soybean',
		2: 'maize',
		3: 'cotton',
		4: 'coffee',
		5: 'beans',
		6: 'sorghum',
		7: 'millet',
		8: 'eucalyptus',
		9: 'pasture',
		10: 'hay',
		11: 'cerrado',
		12: 'conversion_area',
		13: 'soil',
		14: 'not_identified'}
	def load_label(label_list,
		label_path):
		label = np.zeros((len(label_list), trainTest.data.mask.shape[0],
			trainTest.data.mask.shape[1]))
		for idx, label_name in enumerate(label_list):
			# ic(label_path + label_name)
			im = cv2.imread(label_path + label_name + '.tif', -1)
			# ic(np.unique(im, return_counts=True))
			label[idx] = im.copy()
			# ic(np.unique(label[idx], return_counts=True))
			# pdb.set_trace()
		return label
	
	label_path = 'E:/Jorge/osss-mcr/dataset/lm_data/labels/'
	label = load_label(label_list, label_path)
	label = label[-1]
	
	mask_flat = trainTest.data.mask.flatten()
	label_flat = label.flatten()
	features = trainTest.data.full_ims_test
	features = np.transpose(features, (1, 2, 3, 0))
	features = np.reshape(features, (features.shape[0], features.shape[1], -1))
	
	features_flat = np.zeros((features.shape[0] * features.shape[1], features.shape[-1]))
	for chan in range(features.shape[-1]):
		features_flat[..., chan] = features[..., chan].flatten()


	# test 
	mask_test = mask_flat[mask_flat == 2]
	features_test = features_flat[mask_flat == 2]
	label_flat = label_flat[mask_flat == 2]

	idxs = np.random.choice(range(features_test.shape[0]), size = 2000)
	features_test = features_test[idxs]
	label_flat = label_flat[idxs]

	ic(mask_test.shape, features_test.shape)
#	pdb.set_trace()
	time_start = time.time()
#	tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
	tsne = TSNE(n_components=2, verbose=1, perplexity=60, n_iter=300)

	tsne_results = tsne.fit_transform(features_test)
	print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


	df_subset = {}
	df_subset['tsne-2d-one'] = tsne_results[:,0]
	df_subset['tsne-2d-two'] = tsne_results[:,1]
#	df_subset['label'] = label_flat
	df_subset['label'] = []

	ic(np.unique(label_flat, return_counts = True))

	for label_id in range(label_flat.shape[0]):

		df_subset['label'].append(class_names[label_flat[label_id]])

	plt.figure(figsize=(16,10))
	sns.scatterplot(
		x="tsne-2d-one", y="tsne-2d-two",
		hue="label",
		palette=sns.color_palette("hls", len(np.unique(label_flat))),
		data=df_subset,
		legend="full"
#		alpha=0.3
	)
	plt.show()
	pdb.set_trace()


	



