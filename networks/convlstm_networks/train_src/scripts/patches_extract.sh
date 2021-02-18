dataset=$1 # could be cv or lem
#dataset_source='SAR' # could be SAR or Optical
#dataset_source='Optical' # could be SAR or Optical
dataset_source=$2
# ==== EXTRACT PATCHES

cd ../../../../dataset/dataset/patches_extract_script

#python patches_store.py -ttmn="TrainTestMask.tif" -mm="ram" --debug=1 -pl 32 -po 0 -ts 0 -tnl 10000 -bs=5000 --batch_size=128 --filters=256 -m="smcnn_semantic" --phase="repeat" -sc=False --class_n=$class_n --log_dir="../data/summaries/" --path="${dataset_path}" --im_h=$im_h --im_w=$im_w --band_n=2 --t_len=$t_len --id_first=1 -tof=False -nap=10000 -psv=True
##python patches_store.py -mm="ram" --debug=1 -pl 32 -po 0 -ts 0 -tnl 10000 -bs=5000 --batch_size=128 --filters=256 -m="smcnn_semantic" --phase="repeat" -sc=False --class_n=$class_n --log_dir="../data/summaries/" --path="${dataset_path}" --im_h=$im_h --im_w=$im_w --band_n=2 --t_len=$t_len --id_first=1 -tof=False -nap=10000 -psv=True
python patches_store.py --dataset_source=$dataset_source --dataset_name=$dataset -mm="ram" --debug=1 -pl 32 -po 0 -ts 0 -tnl 10000 -bs=5000 --batch_size=128 --filters=256 -m="smcnn_semantic" --phase="repeat" -sc=False --log_dir="../data/summaries/" --id_first=1 -tof=False -nap=10000 -psv=True

cd ../../../networks/convlstm_networks/train_src/scripts




