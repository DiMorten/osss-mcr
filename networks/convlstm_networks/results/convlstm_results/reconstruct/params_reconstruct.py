
class ParamsReconstruct():
    def __init__(self):
        self.prediction_type = 'model'
        self.save_input_im = True

        self.croppedFlag = False
        self.open_set_mode = True
        self.mosaic_flag = True

        self.threshold_idx = 4
        self.overlap = 0.5

        self.add_padding_flag = True

#        self.overlap_mode = 'average' # average, replace
        self.overlap_mode = 'replace' # average, replace



