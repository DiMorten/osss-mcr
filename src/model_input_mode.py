import deb
import numpy as np
import pdb
class ModelInputMode():
    pass
class MIMFixed(ModelInputMode):
    def __init__(self):
        pass
    def batchTrainPreprocess(self, batch, data, label_date_id, batch_seq_len=12):
        input_ = batch['in'].astype(np.float16)
        input_ = data.addDoty(input_)
        return input_
    def batchValPreprocess(self, batch, data):
        input_ = ['in'].astype(np.float16)
        input_ = data.addDoty(input_)
        return input_

    def trainingInit(self,batch,data,t_len, model_t_len):
        batch['train']['shape'] = (batch['train']['size'], model_t_len) + data.patches['train']['in'].shape[2:]
        batch['val']['shape'] = (batch['val']['size'], model_t_len) + data.patches['val']['in'].shape[2:]
        batch['test']['shape'] = (batch['test']['size'], model_t_len) + data.patches['test']['in'].shape[2:]

        deb.prints(batch['train']['shape'])
        data.ds.dotyReplicateSamples()
        #data.labeled_dates = 12
##        deb.prints(data.labeled_dates)
#        min_seq_len = t_len - data.labeled_dates + 1 # 20 - 12 + 1 = 9
#        deb.prints(min_seq_len)

        return batch, data, None

    def valLabelSelect(self, data, label_id = -1):
        return data
class MIMFixed_PaddedSeq(MIMFixed):
    def batchTrainPreprocess(self, batch, ds, label_date_id, batch_seq_len=12):
        len_input_seq = batch['in'].shape[1]
        #print("batch shape, len input seq", batch['shape'], len_input_seq)
        if len_input_seq<batch_seq_len:
            input_ = np.zeros(batch['shape']).astype(np.float16)
            input_[:, -len_input_seq:] = batch['in']
            input_ = ds.addDotyPadded(input_, 
                        bounds = None, 
                        seq_len = batch_seq_len,
                        sample_n =  batch['in'].shape[0])
            return input_
        else:
            input_ = batch['in'].astype(np.float16)
            input_ = ds.addDoty(input_)
            return input_

    def trainPreprocess(self, full_ims_train, ds, label_date_id, batch_seq_len=12):
        len_input_seq = full_ims_train.shape[0]
        #print("batch shape, len input seq", batch['shape'], len_input_seq)
        if len_input_seq<batch_seq_len:
            full_ims_train_seq_padded = np.zeros(
                (len_input_seq,*full_ims_train.shape[1:])).astype(np.float16)
            full_ims_train_seq_padded[:, -len_input_seq:] = full_ims_train
            full_ims_train_seq_padded = ds.addDotyPadded(input_, 
                        bounds = None, 
                        seq_len = batch_seq_len,
                        sample_n =  batch['in'].shape[0])
            return input_
        else:
            input_ = batch['in'].astype(np.float16)
            input_ = ds.addDoty(input_)
            return input_

class MIMFixedLabelSeq(MIMFixed):
    def __init__(self):
        pass
    def batchTrainPreprocess(self, batch, data, label_date_id, batch_seq_len=12):
        input_ = batch['train']['in'][:,:label_date_id].astype(np.float16)
        input_ = data.addDoty(input_)
        return input_
    def batchValPreprocess(self, batch, data):
        input_ = batch['val']['in'].astype(np.float16)
        input_ = data.addDoty(input_)
        return input_

class MIMVariable(ModelInputMode):

    def trainingInit(self,batch,data,t_len, model_t_len):
        batch['train']['shape'] = (batch['train']['size'], model_t_len) + data.patches['train']['in'].shape[2:]
        batch['val']['shape'] = (batch['val']['size'], model_t_len) + data.patches['val']['in'].shape[2:]
        batch['test']['shape'] = (batch['test']['size'], model_t_len) + data.patches['test']['in'].shape[2:]

        deb.prints(batch['train']['shape'])
        #data.labeled_dates = 12
        deb.prints(data.labeled_dates)
        min_seq_len = t_len - data.labeled_dates + 1 # 20 - 12 + 1 = 9
        deb.prints(min_seq_len)
        data.ds.setDotyFlag(True)
        return batch, data, min_seq_len
    def valLabelSelect(self, data, label_id = -1):
        
        data.patches['val']['label'] = data.patches['val']['label'][:, label_id]
        data.patches['test']['label'] = data.patches['test']['label'][:, label_id]
        deb.prints(data.patches['val']['label'].shape)

        deb.prints(data.patches['test']['label'].shape)
        return data

class MIMVarLabel(MIMVariable):
    def __init__(self):
        self.batch_seq_len = 12
        pass
    def batchTrainPreprocess(self, batch, data, label_date_id, batch_seq_len=12):
        
        #print("Label, seq start, seq end",label_date_id,label_date_id-batch_seq_len+1,label_date_id+1)
        if label_date_id+1!=0:

            input_ = batch['in'][:, label_date_id-batch_seq_len+1:label_date_id+1]
        else:
            input_ = batch['in'][:, label_date_id-batch_seq_len+1:]
            #print("exception", input_.shape)
        #print(input_.shape)
        #print(label_date_id, batch_seq_len, label_date_id-batch_seq_len+1, label_date_id+1)
        #pdb.set_trace()
        input_ = input_.astype(np.float16)
        input_ = data.addDoty(input_, 
                    bounds = [label_date_id-batch_seq_len+1, label_date_id+1])
        return input_
    def batchMetricSplitPreprocess(self, batch, data, label_date_id, batch_seq_len=12):
        return batchTrainPreprocess(batch, data, label_date_id, batch_seq_len)

class MIMFixedLabelAllLabels(MIMVarLabel):

    def valLabelSelect(self, data, label_id = -1):
        return data

class MIMVarLabel_PaddedSeq(MIMVarLabel):
    def batchTrainPreprocess(self, batch, ds, label_date_id, batch_seq_len=None):
        sample_n = batch['in'].shape[0]
        #print("Label, seq start, seq end",label_date_id,label_date_id-batch_seq_len+1,label_date_id+1)
        if label_date_id+1!=0:
            if label_date_id in ds.padded_dates:
                unpadded_input = batch['in'][:, :label_date_id+1]
                len_input_seq = unpadded_input.shape[1]
                #deb.prints(len_input_seq)
                input_ = np.zeros(batch['shape']).astype(np.float16)
                input_[:, -len_input_seq:] = unpadded_input
            else:
                #print(batch['in'].shape,label_date_id-self.batch_seq_len+1,label_date_id+1)
                input_ = batch['in'][:, label_date_id-self.batch_seq_len+1:label_date_id+1]
                ##print(input_.shape)

        else:
            #print(batch['in'].shape,label_date_id-self.batch_seq_len+1,label_date_id+1)
            input_ = batch['in'][:, label_date_id-self.batch_seq_len+1:]
            ##print(input_.shape)

            #print("exception", input_.shape)
        input_ = input_.astype(np.float16)
        input_ = ds.addDotyPadded(input_, 
                    bounds = [label_date_id-self.batch_seq_len+1, label_date_id+1], 
                    seq_len = self.batch_seq_len,
                    sample_n = sample_n)
        #print(len(input_), input_[0].shape, input_[1].shape)
        
        return input_
    def batchMetricSplitPreprocess(self, batch, data, split='val'):
        input_ = batch[split]['in'][:,-12:].astype(np.float16)
        input_ = data.addDoty(input_, bounds=[-12, None])
        return input_
    # to do: replace batchMetricSplitPreprocess by iteration of all 12 labels,
    # including padded first input sequences.
    def valLabelSelect(self, data, label_id = -1):
        return data


class MIMVarSeqLabel(MIMVariable):
    def __init__(self):
        pass
    def batchTrainPreprocess(self, batch, data, label_date_id, batch_seq_len, t_len):

        # self.t_len is 20 as an example 
        ##label_date_id = np.random.randint(-data.labeled_dates,0) # labels can be from -1 to -12
        # example: last t_step can use entire sequence: 20 + (-1+1) = 20
        # example: first t_step can use sequence: 20 + (-12+1) = 9
        # to do: add sep17 image 
        max_seq_len = t_len + (label_date_id+1) # from 9 to 20
        
        if min_seq_len == max_seq_len:
            batch_seq_len = min_seq_len
        else:
            batch_seq_len = np.random.randint(min_seq_len,max_seq_len+1) # from 9 to 20 in the largest case

        # example: -1-20+1:-1 = -20:-1
        # example: -12-9+1:-12 = -20:-12
        # example: -3-11+1:-3 = -13:-3 
        # example: -1-18+1:-1+1 = -18:0
        ##print("Batch slice",label_date_id-batch_seq_len+1,label_date_id+1)
        ##deb.prints(label_date_id+1!=0)
        ##deb.prints(label_date_id)
        if label_date_id+1!=0:
            batch['train']['in'] = batch['train']['in'][:, label_date_id-batch_seq_len+1:label_date_id+1]
        else:
            batch['train']['in'] = batch['train']['in'][:, label_date_id-batch_seq_len+1:]

        #deb.prints(batch['train']['in'].shape[1])
        #deb.prints(batch['train']['in'].shape[1] == batch_seq_len)
        #deb.prints(batch_seq_len)
        #deb.prints(label_date_id)
        assert batch['train']['in'].shape[1] == batch_seq_len

        input_ = np.zeros(batch['train']['shape']).astype(np.float16)
        input_[:, -batch_seq_len:] = batch['train']['in']
        input_ = data.addDoty(input_)
        return input_
