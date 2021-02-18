 
import numpy as np
import cv2
import glob
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('-ds', '--dataset', dest='dataset',
					default='lm', help='t len')
parser.add_argument('-mdl', '--model', dest='model',
					default='densenet', help='t len')

a = parser.parse_args()

dataset=a.dataset
model=a.model
postprocessing='viterbi'
path='/home/lvc/Jorg/igarss/convrnn_remote_sensing/results/seq2seq_ignorelabel/'


#dataset='cv'
#model='biconvlstm'
#model='densenet'
#model='convlstm'
label_path=dataset+'/convlstm/'

#model_out_path=dataset+'/'+model+'/'
model_out_path="results_softmax_"+postprocessing+"/"+dataset+'/'+model+'/'
if dataset=='lm':
	data_path='../../../../../deep_learning/LSTM-Final-Project/'
	path+='lm/'
	predictions_path=path+'prediction_DenseNetTimeDistributed_128x2_batch16_full.npy'
	mask_path=data_path+'lm_data/TrainTestMask.tif'
	location_path=data_path+'src_seq2seq/locations/lm/'
	folder_load_path=data_path+'lm_data/train_test/test/labels/'

	custom_colormap = np.array([[255,146,36],
					[255,255,0],
					[164,164,164],
					[255,62,62],
					[0,0,0],
					[172,89,255],
					[0,166,83],
					[40,255,40],
					[187,122,83],
					[217,64,238],
					[0,113,225],
					[128,0,0],
					[114,114,56],
					[53,255,255]])
	t_steps=[0,7,11]
	position=[int(8454/1.35),2770] #int(8658/4)
	print(position)
	delta=[880,450]
elif dataset=='cv':
	data_path='../../../../../deep_learning/LSTM-Final-Project/'

	path+='cv/'
	predictions_path=path+'prediction_DenseNetTimeDistributed_128x2_batch16_full.npy'
	mask_path=data_path+'cv_data/TrainTestMask.tif'
	location_path=data_path+'src_seq2seq/locations/cv/'

	folder_load_path=data_path+'cv_data/train_test/test/labels/'

	custom_colormap = np.array([[255, 146, 36],
                   [255, 255, 0],
                   [164, 164, 164],
                   [255, 62, 62],
                   [0, 0, 0],
                   [172, 89, 255],
                   [0, 166, 83],
                   [40, 255, 40],
                   [187, 122, 83],
                   [217, 64, 238],
                   [45, 150, 255]])
	t_steps=[0,4,10]
	position=[6517,1000]
	print(position)
	delta=[767,1800]
labels=np.load(path+'labels.npy').argmax(axis=4)


# label0=cv2.imread(dataset+model+'/label_cv_t0.png')
# prediction0=cv2.imread(dataset+model+'/prediction_cv_t0.png')

# label4=cv2.imread(dataset+model+'/label_cv_t4.png')
# prediction4=cv2.imread(dataset+model+'/prediction_cv_t4.png')

# label7=cv2.imread(dataset+model+'/label_cv_t10.png')
# prediction7=cv2.imread(dataset+model+'/prediction_cv_t10.png')
print("LOading labels....")
print(model_out_path+'label_cv_t0_'+model+'.png')


def im_read_crop_store(im_path,out_path,position,delta,dataset):
	print(im_path)
	im=cv2.imread(im_path)
	print(im.shape)
	out=im[position[0]:position[0]+delta[0],position[1]:position[1]+delta[1]]
	if dataset=='lm':
		mode=2
		if mode==1:
			out=np.concatenate((out[:350,:],out[500:,:]),axis=0)
		elif mode==2:
			A=im[6382:6612,2985:3188]
			B=im[6825:7130,2744:3064]
			C=im[1468:1648,4358:4578]
			C=np.rot90(C)
			out=np.ones((540,400,3))*255

			def im_fill_with_cropped(out,im,cropped,row,col):
				#print(row,row+cropped.shape[0],col,col+cropped.shape[1])
				out[row:row+cropped.shape[0],col:col+cropped.shape[1]]=cropped.copy()
				return out
			#out=im_fill_with_cropped(out,im,A,0,205)
			#out=im_fill_with_cropped(out,im,C,0,0)
			
			#out=im_fill_with_cropped(out,im,B,250,0)
			
			out=im_fill_with_cropped(out,im,A,0,0)
			out=im_fill_with_cropped(out,im,C,0,190)
			out=im_fill_with_cropped(out,im,B,225,30)
				
			polys=[
				np.array([[119,0],[190,0],[190,138]]),
				np.array([[104,223],[162,218],[200,319],[161,337]]),
				np.array([[317,308],[350,308],[357,357]])
			]
			cv2.fillPoly(out,pts=polys,color=(255,255,255))
	elif dataset=='cv':
		mode=2
		if mode==1:
			outA=out[:,:949]
			outB=out[:,949:]
			out=np.ones((out.shape[0]*2,outA.shape[1],3))*255 # pass right to bottom
			out[:outA.shape[0],:outA.shape[1],:]=outA.copy()
			out[outA.shape[0]:,50:50+outB.shape[1],:]=outB.copy()
			out=out[:-150,:] # shorten lower im
		elif mode==2:
			out=np.ones((845,680,3))*255
			A=im[6496:6899,2540:2751]
			B=im[6861:7289,1527:1955]
			C=im[6520:6954,992:1398]
			D=im[6913:7114,1999:2249]
			E=im[6888:7032,1377:1525]
			print(A.shape,B.shape,C.shape,D.shape,E.shape,out.shape)
			print(np.average(C),np.average(B))
			def im_fill_with_cropped(out,im,cropped,row,col):
				#print(row,row+cropped.shape[0],col,col+cropped.shape[1])
				out[row:row+cropped.shape[0],col:col+cropped.shape[1]]=cropped.copy()
				return out
			#out=im_fill_with_cropped(out,im,C,0,0)
			#out=im_fill_with_cropped(out,im,B,0,407)
			#out=im_fill_with_cropped(out,im,A,430,460)
			#out=im_fill_with_cropped(out,im,D,450,30)
			#out=im_fill_with_cropped(out,im,E,660,220)


			out=im_fill_with_cropped(out,im,C,380,250)
			out=im_fill_with_cropped(out,im,B,0,250)
			out=im_fill_with_cropped(out,im,A,0,20)
			out=im_fill_with_cropped(out,im,D,430,0)
			out=im_fill_with_cropped(out,im,E,660,30)

			#out[0:C.shape[0],0:C.shape[1]]=C.copy()
			#outA=out[:,:949]
			#outB=out[:,949:]
			#out=np.ones((out.shape[0]*2,outA.shape[1],3))*255 # pass right to bottom
			#out[:outA.shape[0],:outA.shape[1],:]=outA.copy()
			#out[outA.shape[0]:,50:50+outB.shape[1],:]=outB.copy()
			#out=out[:-150,:] # shorten lower im
		
	cv2.imwrite(out_path,out)


def ims_read_crop_store(im_id_list,model_out_path,
	model,position,delta,dataset,label_path):
	for im_id in im_id_list:
		im_path=label_path+'label_t'+str(im_id)+'_convlstm.png'
		out_path=model_out_path+'point_of_interest/label_t'+str(im_id)+'.png'
		im_read_crop_store(im_path,out_path,position,delta,dataset)
		im_path=model_out_path+'prediction_t'+str(im_id)+'_'+model+'.png'
		out_path=model_out_path+'point_of_interest/prediction_t'+str(im_id)+'.png'
		im_read_crop_store(im_path,out_path,position,delta,dataset)

#if dataset=='cv':
#elif dataset=='lm':


ims_read_crop_store(t_steps,model_out_path,model,
	position,delta,dataset,label_path)

