@echo off

::KERAS_BACKEND=tensorflow
::id='sarh_tvalue20'
::id='sarh_tvalue40fixed'
::id='sarh_tvalue20repeat'
:: set id=windows_test
:: set id=int16_adagrad_crossentropy


::dataset=cv
:: set dataset=l2
set dataset=l2
set model_dataset=l2

::::dataSource='OpticalWithClouds'
::dataSource='SAR'
set dataSource=SAR
set model=UUnet4ConvLSTM_doty
:: set seq_mode=fixed
set seq_mode=fixed

:: cd ../analysis/

:: cd ../../../train_src/scripts
:: call patches_extract.bat %dataset% %dataSource% %seq_mode% %seq_date%
cd ../analysis/
:: python analysis_nto1_fixedseq_fixedlabel.py --dataset=%dataset% --seq_date=%seq_date%


set seq_date=apr
set id=fixed_label_%seq_mode%_%seq_date%_l2_secondtry
python analysis_nto1_fixedseq_fixedlabel.py --dataset=%dataset% --model_dataset=%model_dataset% --seq_date=%seq_date%


:: ===== USE MODEL
::. experiment_automation.sh $id 'BUnet4ConvLSTM_SkipLSTM' $dataset
::. experiment_automation.sh $id 'Unet3D' $dataset
::. experiment_automation.sh $id 'BUnet4ConvLSTM_64' $dataset  :: Unet5 uses 1 conv. in



::. experiment_automation.sh $id 'BUnet4ConvLSTM' $dataset $dataSource  :: Unet5 uses 1 conv. in
::. experiment_automation.sh $id 'ConvLSTM_seq2seq' $dataset  :: Unet5 uses 1 conv. in
::. experiment_automation.sh $id 'ConvLSTM_seq2seq_bi' $dataset
::. experiment_automation.sh $id 'BAtrousGAPConvLSTM' $dataset  :: gonna test balancing after replication
::. experiment_automation.sh $id 'DenseNetTimeDistributed_128x2' $dataset  :: Unet5 uses 1 conv. in

