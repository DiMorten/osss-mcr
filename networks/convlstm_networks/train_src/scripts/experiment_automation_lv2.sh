KERAS_BACKEND=tensorflow
#id='sarh_tvalue20'
#id='sarh_tvalue40fixed'
id='sarh_tvalue20repeat'
id='sar_bdenseconvlstm_check128'


#dataset='cv'
dataset='cv'
##dataSource='OpticalWithClouds'
#dataSource='SAR'
dataSource='SAR'

# ==== EXTRACT PATCHES
. patches_extract.sh $dataset $dataSource
# ===== USE MODEL
#. experiment_automation.sh $id 'BUnet4ConvLSTM_SkipLSTM' $dataset
#. experiment_automation.sh $id 'Unet3D' $dataset
#. experiment_automation.sh $id 'BUnet4ConvLSTM_64' $dataset  # Unet5 uses 1 conv. in



. experiment_automation.sh $id 'BUnet4ConvLSTM' $dataset $dataSource  # Unet5 uses 1 conv. in
#. experiment_automation.sh $id 'ConvLSTM_seq2seq' $dataset  # Unet5 uses 1 conv. in
#. experiment_automation.sh $id 'ConvLSTM_seq2seq_bi' $dataset
#. experiment_automation.sh $id 'BAtrousGAPConvLSTM' $dataset  # gonna test balancing after replication
#. experiment_automation.sh $id 'DenseNetTimeDistributed_128x2' $dataset  # Unet5 uses 1 conv. in

