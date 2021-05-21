@echo off

::KERAS_BACKEND=tensorflow
::id='sarh_tvalue20'
::id='sarh_tvalue40fixed'
::id='sarh_tvalue20repeat'
:: set id=windows_test
:: set id=int16_adagrad_crossentropy


:: set dataset=lm
set dataset=lm
:: set dataset=lm

::::dataSource='OpticalWithClouds'
::dataSource='SAR'
set dataSource=SAR
set model=UUnet4ConvLSTM
:: set model=UUnet4ConvLSTM_doty

set seq_mode=fixed



set loco_class=8
:: pasture

set seq_date=mar
:: set seq_date=jun

:: set seq_date=feb

:: set id=fixed_label_%seq_mode%_%seq_date%_lm_testlm_fewknownclasses_valrand_dummy
:: set id=fixed_label_%seq_mode%_%seq_date%_loco%loco_class%_lm_testlm_lessclass8_groupclasses
:: set id=fixed_label_%seq_mode%_%seq_date%_lm_fewknownclasses2
:: set id=len6_%seq_date%
set id=dummy_fit

:: set id=fixed_label_%seq_mode%_%seq_date%_lm_testlm_2kkc
:: set id=fixed_label_%seq_mode%_%seq_date%_lm_testlm_2kkc_groupclasses
:: set id=fixed_label_%seq_mode%_%seq_date%_lm_testlm_allkkc

:: call patches_extract.bat %dataset% %dataSource% %seq_mode% %seq_date%
call experiment_automation.bat %id% %model% %dataset% %dataSource% %seq_mode% %seq_date% %loco_class%

:: set seq_date=jul
:: set id=fixed_label_%seq_mode%_%seq_date%_l2
:: call patches_extract.bat %dataset% %dataSource% %seq_mode% %seq_date%
:: call experiment_automation.bat %id% %model% %dataset% %dataSource% %seq_mode% %seq_date%
:: ===== USE MODEL
::. experiment_automation.sh $id 'BUnet4ConvLSTM_SkipLSTM' $dataset
::. experiment_automation.sh $id 'Unet3D' $dataset
::. experiment_automation.sh $id 'BUnet4ConvLSTM_64' $dataset  :: Unet5 uses 1 conv. in



::. experiment_automation.sh $id 'BUnet4ConvLSTM' $dataset $dataSource  :: Unet5 uses 1 conv. in
::. experiment_automation.sh $id 'ConvLSTM_seq2seq' $dataset  :: Unet5 uses 1 conv. in
::. experiment_automation.sh $id 'ConvLSTM_seq2seq_bi' $dataset
::. experiment_automation.sh $id 'BAtrousGAPConvLSTM' $dataset  :: gonna test balancing after replication
::. experiment_automation.sh $id 'DenseNetTimeDistributed_128x2' $dataset  :: Unet5 uses 1 conv. in

