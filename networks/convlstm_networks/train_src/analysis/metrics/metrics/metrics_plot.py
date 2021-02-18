import numpy as np
from pathlib import Path

path=Path('metrics/')
exp_names=['metrics_model_best_BUnet4ConvLSTM_sar_testingbaseline.h5.npy',
    'metrics_model_best_BUnet4ConvLSTM_sarh_tvalue20.h5.npy',
    'metrics_model_best_BUnet4ConvLSTM_sarh_tvalue20repeat.h5.npy',
    'metrics_model_best_BUnet4ConvLSTM_sarh_tvalue40fixed.h5.npy',
    'metrics_model_best_BUnet4ConvLSTM_float32.h5.npy',
    'metrics_model_best_BUnet4ConvLSTM_int16.h5.npy']


for exp_name in exp_names:
    metrics=np.expand_dims(np.load(path/exp_name, allow_pickle=True),axis=-1)[0]
    print(exp_name)
    print('f1: ',np.average(metrics['f1_score']), ', oa: ',np.average(metrics['overall_acc']))
    print(metrics['f1_score'])
#a=
#print(a)