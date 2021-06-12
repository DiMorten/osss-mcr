:: @echo off
set id=%1
set model=%2
set dataset=%3 
:: could be cv or lem
set dataSource=%4
set seq_mode=%5
set seq_date=%6

:: open set
set loco_class=%7

echo %dataset%
echo 'cv' 
IF %dataset%==cv (
	set filename=campo_verde
	set results_path=../results/convlstm_results/model/cv/
	set summary_save_path=../results/convlstm_results/summary/cv/
	set dataset_path=../../../dataset/dataset/cv_data/
	set sequence_len=14
	set class_n=12
	set channel_n=2
) ELSE (
	IF %dataset%==lm (
		set filename=lm
		set results_path=../results/convlstm_results/model/lm/
		set summary_save_path=../results/convlstm_results/summary/lm/
		set dataset_path=../../../dataset/dataset/lm_data/
		set sequence_len=19

		::sequence_len=11
		set class_n=15 
		IF %dataSource%==SARH (
			set channel_n=3
			set results_path=../results/convlstm_results/model/lm_sarh/
			set summary_save_path=../results/convlstm_results/summary/lm_sarh/
		) ELSE (
			set channel_n=2
		)
	) ELSE (
		set filename=l2
		set results_path=../results/convlstm_results/model/l2/
		set summary_save_path=../results/convlstm_results/summary/l2/
		set dataset_path=../../../dataset/dataset/l2_data/
		set sequence_len=19

		set class_n=15 
		set channel_n=2
	)
)
echo %results_path%
echo %channel_n%





set stop_epoch=400 
:: promote to lv2?

:: ::id="blockgoer"
:: rm -f log1.txt
:: rm -f log2.txt
:: rm -f log3.txt
:: ::model='FCN_ConvLSTM'
:: ::::model='ConvLSTM_DenseNet'
:: ::model='FCN_ConvLSTM2'
:: ::model='BiConvLSTM_DenseNet'
:: ::::model='ConvLSTM_seq2seq'
:: ::::model='FCN_ConvLSTM_seq2seq_bi'
:: ::::model='FCN_ConvLSTM_seq2seq_bi_skip'

:: ::::model='DenseNetTimeDistributed'
:: ::model='ConvLSTM_seq2seq_bi' :: russworm bi .
:: :: ============== EXECUTE EXPERIMENT ===============
cd ..
python main_legacy_patches.py --stop_epoch=%stop_epoch% -pl=32 -pstr=32 -psts=32 -bstr=16 -bsts=16 -path=%dataset_path% -tl=%sequence_len% -cn=%class_n% -chn=%channel_n% -mdl=%model% -ds=%dataset% --seq_mode=%seq_mode% --seq_date=%seq_date% --id=%id% --loco_class=%loco_class%
:: ::python main_hdd.py -pl=32 -pstr=32 -psts=32 -path=$dataset_path -tl=$sequence_len -cn=$class_n -chn=2 -mdl=$model
:: echo %filename%_%model%_%id%

:: :: ========= TAKE SCREENSHOT ===============
:: im_name="${filename}_${model}_${id}.png"
:: wmctrl -a konsole
:: shutter -f -o $im_name -e

:: :: ============== SEND IMAGE TO FACEBOOK MESSENGER =========
:: cd scripts
:: path="../${im_name}"
:: echo "${path}"
:: . ifttt_send.sh $path
::cd -
:: =============== MOVE PREDICTIONS TO RESULT FOLDER ======
::results_path='../results/seq2seq_ignorelabel/cv/'

echo %results_path%model_best_%model%_%id%.h5 

set word=\
call set results_path=%%results_path:/=%word%%%
echo %results_path%model_best_%model%_%id%.h5 

echo F|xcopy model_best.h5 %results_path%model_best_%model%_%id%.h5 /f /y
 

set word=\
call set summary_save_path=%%summary_save_path:/=%word%%%
echo %summary_save_path%summary_%model%_%id%.txt

echo F|xcopy model_summary.txt %summary_save_path%summary_%model%_%id%.txt /f /y
cd scripts

