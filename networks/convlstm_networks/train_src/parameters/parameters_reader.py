import os
import json
from icecream import ic
import pdb

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):

        assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__

class ParamsTrain(Params):
    def __init__(self, folder_path):
        # 4 known classes
        json_path = folder_path+'parameters_openset.json'
#        json_path = folder_path+'parameters_closedset_groupclasses.json'
        json_path = folder_path+'save_nonaugmented_train_patches.json'

        # 2 known classes
#        json_path = folder_path+'twokkc_parameters_openset.json'
#        json_path = folder_path+'twokkc_parameters_closedset_groupclasses.json'
#        json_path = folder_path+'twokkc_save_nonaugmented_train_patches.json'

        # all kkc
#        json_path = folder_path+'allkkc_parameters_openset.json'
#        json_path = folder_path+'allkkc_parameters_closedset_groupclasses.json'
#        json_path = folder_path+'allkkc_save_nonaugmented_train_patches.json'

#        json_path = folder_path+'parameters_openset_specifyunknownclasses.json'
#        json_path = folder_path+'save_nonaugmented_train_patches_unknownclasses.json'

#        json_path = folder_path+'parameters_openset_lessclass8.json'
#        json_path = folder_path+'save_nonaugmented_train_patches_lessclass8.json'
#        json_path = folder_path+'parameters_closedset_groupclasses_lessclass8.json'

        # 3 known classes campo verde
#        json_path = folder_path+'cv/'+'parameters_openset.json'
#        json_path = folder_path+'parameters_closedset_groupclasses.json'
#        json_path = folder_path+'cv/'+'save_nonaugmented_train_patches.json'

        print(os.listdir(folder_path))
        super().__init__(json_path)
        ic(self.seq_date)
#        pdb.set_trace()
        if self.dataset == 'lm':
            if self.seq_date == 'mar':
                self.known_classes = [0, 1, 10, 12]
            elif self.seq_date == 'jun':
                self.known_classes = [1, 6, 10, 12]
        elif self.dataset == 'cv':
            if self.seq_date == 'jun':
                self.known_classes = [1, 2, 8]

        print("self.known_classes", self.known_classes)
#        pdb.set_trace()
class ParamsAnalysis(Params):
    def __init__(self, folder_path):

        json_path = folder_path+'parameters_analysis_closedset.json'

        print(os.listdir(folder_path))
        super().__init__(json_path)