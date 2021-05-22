@echo off

::KERAS_BACKEND=tensorflow
::id='sarh_tvalue20'
::id='sarh_tvalue40fixed'
::id='sarh_tvalue20repeat'
:: set id=windows_test
:: set id=int16_adagrad_crossentropy


::dataset=cv
:: set dataset=l2
set dataset=lm
set model_dataset=lm
::::dataSource='OpticalWithClouds'
::dataSource='SAR'
set dataSource=SAR
set model=UUnet4ConvLSTM
:: set seq_mode=fixed
set seq_mode=fixed



cd ../analysis/
:: set seq_date=mar
set seq_date=mar

set id=fixed_label_%seq_mode%_%seq_date%_lm_firsttry
:: python analysis_nto1_fixedseq_fixedlabel.py --dataset=%dataset% --model_dataset=%model_dataset% --seq_date=%seq_date%
:: python analysis_nto1_fixedseq_fixedlabel_nounknown.py --dataset=%dataset% --model_dataset=%model_dataset% --seq_date=%seq_date%
python analysis_nto1_fixedseq_fixedlabel_groupclasses_closed.py --dataset=%dataset% --model_dataset=%model_dataset% --seq_date=%seq_date%

