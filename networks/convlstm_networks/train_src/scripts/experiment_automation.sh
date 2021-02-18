
id=$1
model=$2
dataset=$3 # could be cv or lem
dataSource=$4

if [ "$dataset" == "cv_seq1" ]
then
	filename="campo_verde"
	results_path='../results/convlstm_results/model/cv/'
	summary_save_path='../results/convlstm_results/summary/cv/'
	dataset_path="../../../dataset/dataset/cv_data/"
	sequence_len=7
	class_n=12
elif [ "$dataset" == "cv" ]
then
	filename="campo_verde"
	results_path='../results/convlstm_results/model/cv/'
	summary_save_path='../results/convlstm_results/summary/cv/'
	dataset_path="../../../dataset/dataset/cv_data/"
	sequence_len=14
	class_n=12
	channel_n=2

else
	filename="lm"
	results_path='../results/convlstm_results/model/lm/'
	summary_save_path='../results/convlstm_results/summary/lm/'
	dataset_path="../../../dataset/dataset/lm_data/"
	sequence_len=13
	#sequence_len=11
	class_n=15 # 14+bcknd

	if [ "$dataSource" == "SARH" ]
	then
		channel_n=3
		results_path='../results/convlstm_results/model/lm_sarh/'
		summary_save_path='../results/convlstm_results/summary/lm_sarh/'

	elif [ "$dataSource" == "SAR" ]
	then
		channel_n=2
	else
		channel_n=2 #optical missing
	fi	
fi


stop_epoch=1 # promote to lv2?

# #id="blockgoer"
# rm -f log1.txt
# rm -f log2.txt
# rm -f log3.txt
# #model='FCN_ConvLSTM'
# ##model='ConvLSTM_DenseNet'
# #model='FCN_ConvLSTM2'
# #model='BiConvLSTM_DenseNet'
# ##model='ConvLSTM_seq2seq'
# ##model='FCN_ConvLSTM_seq2seq_bi'
# ##model='FCN_ConvLSTM_seq2seq_bi_skip'

# ##model='DenseNetTimeDistributed'
# #model='ConvLSTM_seq2seq_bi' # russworm bi .
# # ============== EXECUTE EXPERIMENT ===============
cd ..
python main.py --stop_epoch=$stop_epoch -pl=32 -pstr=32 -psts=32 -bstr=16 -bsts=16 -path=$dataset_path -tl=$sequence_len -cn=$class_n -chn=$channel_n -mdl=$model
# #python main_hdd.py -pl=32 -pstr=32 -psts=32 -path=$dataset_path -tl=$sequence_len -cn=$class_n -chn=2 -mdl=$model
echo "${filename}_${model}_${id}"

# # ========= TAKE SCREENSHOT ===============
# im_name="${filename}_${model}_${id}.png"
# wmctrl -a konsole
# shutter -f -o $im_name -e

# # ============== SEND IMAGE TO FACEBOOK MESSENGER =========
# cd scripts
# path="../${im_name}"
# echo "${path}"
# . ifttt_send.sh $path
#cd -
# =============== MOVE PREDICTIONS TO RESULT FOLDER ======
#results_path='../results/seq2seq_ignorelabel/cv/'
cp model_best.h5 "${results_path}model_best_${model}_${id}.h5"
#cp prediction.npy "${results_path}prediction_${model}_${id}.npy"

cp model_summary.txt "${summary_save_path}summary_${model}_${id}.txt" 
cd scripts

