from dataSource import Dataset,LEM,CampoVerde,DataSource,SARSource
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
class DatasetStats():
    def __init__(self,dataset):
        self.dataset=dataset
    
    def calcAverageTimeseries(self,ims,mask):
        time_delta=self.dataset.getTimeDelta()
        print(time_delta)
        for channel in range(self.dataset.getBandN()):
            averageTimeseries=[]
            for t_step in range(0,self.dataset.t_len):
                im=ims[t_step,:,:,channel]
                #mask_t=mask[t_step]
                
                #print("im shape: {}, mask shape: {}".format(im.shape,mask.shape))
                im=im.flatten()
                mask_t=mask.flatten()
                #print("im shape: {}, mask shape: {}".format(im.shape,mask.shape))

                im=im[mask_t==1] # only train and test pixels (1 and 2)
                averageTimeseries.append(np.average(im))
            averageTimeseries=np.asarray(averageTimeseries)
            plt.figure(channel)
            fig, ax = plt.subplots()
            ax.plot(time_delta,averageTimeseries,marker=".")
            ax.set(xlabel='time ID', ylabel='band',title='Image average over time')
            plt.grid()
            print('averageTimeseries',averageTimeseries)
            plt.show()
    def calcAverageTimeseriesPerClass(self,ims,mask,label):
        print("Label shape",label.shape)
        time_delta=self.dataset.getTimeDelta()
        print(time_delta)
        for channel in range(self.dataset.getBandN()):
            averageTimeseries=[]
            plt.figure(channel)
            fig, ax = plt.subplots()
            ax.set_prop_cycle(cycler('color', ['c', 'm', 'y', 'k','b', 'g', 'r']))

#            for clss,clss_name in zip([1,2,3,9],['soybean','maize','cotton','soil']):
            for clss,clss_name in zip([1,2,3,7,13],['soybean','maize','cotton','millet','soil']):
#            for clss,clss_name in zip(range(self.dataset.getClassN()),self.dataset.getClassList()):
                averageTimeseries=[]
                for t_step in range(0,self.dataset.t_len):
                    # check available classes
                    
                    im=ims[t_step,:,:,channel]
                    label_t=label[t_step]# label is (t,h,w,channel)
                    label_t_unique=np.unique(label_t)
                    if not (clss in label_t_unique):
                        averageTimeseries.append(np.nan)
                        continue
                    #print("Label t shape",label_t.shape)
                    #mask_t=mask[t_step]
                    
                    #print("im shape: {}, mask shape: {}".format(im.shape,mask.shape))
                    im=im.flatten()
                    mask=mask.flatten()
                    label_t=label_t.flatten()
                    #print("im shape: {}, mask shape: {}".format(im.shape,mask.shape))
                    
                    # only train
                    im=im[mask==1]
                    label_t=label_t[mask==1]


                    im=im[label_t==clss] # only train and test pixels (1 and 2) from clss
                    averageTimeseries.append(np.average(im))
                averageTimeseries=np.asarray(averageTimeseries)
                ax.plot(time_delta,averageTimeseries,marker=".",label=clss_name)
                ax.legend()
                print('averageTimeseries',averageTimeseries)
            ax.set(xlabel='time ID', ylabel='band',title='Image average over time')
            plt.grid()
            #
        plt.show()
            



