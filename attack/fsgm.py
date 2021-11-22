import os
import pickle
from utils.functions import makedirs
import numpy as np

class FsgmPickleMaker:
    def __init__(self,dataset):
        pass
    def __load_dataset(self):
        pass
    def __load_model(self):
        pass
    def fsgm(self):
        pass

    def generate(self,save_path,alpha):
        dataset = self.__load_dataset()
        x_data, y_data = dataset['x_data'],dataset['y_data']
        new_x_data = []
        model = self.__load_model()
        for x,y in zip(x_data,y_data):
            gradient, label= model(x)
            if label != y:
                new_x_data.append(x)
                continue
            new_x_data.append(self.fsgm(x,gradient,alpha))
        
        makedirs(save_path)

        dataset['x_data'] = np.array(new_x_data)
        filename = os.path.join(save_path,"emobase2010.pickle")
        with open(filename, 'wb') as handle:
           pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

            