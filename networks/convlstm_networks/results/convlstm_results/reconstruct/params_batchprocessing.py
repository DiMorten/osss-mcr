from icecream import ic
import numpy as np
class ParamsBatchProcessing():
    def __init__(self, paramsTrain, pr):
#        self.batch_processing_n = 3
        if pr.conditionType == 'all':
            if paramsTrain.dataset == 'lm':
                self.batch_processing_n = 30
            elif paramsTrain.dataset == 'cv':   
                self.batch_processing_n = 53
        else:
            if paramsTrain.dataset == 'lm':
                self.batch_processing_n = 1
            elif paramsTrain.dataset == 'cv':
                self.batch_processing_n = 3

            



