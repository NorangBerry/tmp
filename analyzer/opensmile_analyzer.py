import os
import pickle
from utils.functions import normalization_ops
from utils.setting import get_dataset_folder, get_pickle_path
import csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.pyplot import figure

figure(figsize=(8, 6), dpi=300)

class OpenSmileAnalyzer:
    def __init__(self):
        self.label_list:list[str] = []
        self.load_label_list()
        self.feature:str = ""

    def load_label_list(self):
        csv_sample = os.path.join(get_dataset_folder("CREMA-D"),"opensmile","1001_DFA_ANG_XX.csv")
        csvfile = open(csv_sample,'r')
        reader = [each for each in csv.reader(csvfile, delimiter=';')]
        csvfile.close()
        feats_names=str(reader[2:-4]).split(',')[1:-1]
        self.label_list = feats_names

    def process(self,dataset_basename,types=None,values=None):
        #defualt set
        data_path = get_pickle_path(dataset_basename)
        dataset = self.load_pickle_data(data_path)
        result_list = [self.get_data_ranges_from_list(dataset)]
        datasetnames = ["clean"]
        if types is not None:
            for value_list, type in zip(values,types):
                for value in value_list:
                    print(type,value)
                    data_path = get_pickle_path(dataset_basename,type,value)
                    dataset = self.load_pickle_data(data_path)
                    result_list.append(self.get_data_ranges_from_list(dataset))
                    if type == "noisy":
                        datasetnames.append(f"noise({value}dB)")
                    elif type == "gradient":
                        datasetnames.append(f"FGSM(ε={value})")
                    else:
                        datasetnames.append(f"{type}_{value}")
        self.show_plot(result_list,datasetnames)
    
    def load_pickle_data(self,data_path):
        train_filename = f"{data_path}.pickle"
        with open(train_filename, 'rb') as handle:
            data = pickle.load(handle)
            x_data = data["x_data"]
            feat_mu = np.mean(x_data,axis=0)
            feat_st = np.std(x_data, axis=0)
            
            x_data  = normalization_ops(feat_mu, feat_st, x_data)
            return x_data

    def set_target_feature(self,feature:str):
        self.feature = feature

    def get_data_ranges_from_list(self,dataset):
        ret = []
        for data in tqdm(dataset):
            data = self.extract_feature(data)
            if ret == []:
                ret = data
            else:
                ret = np.column_stack((ret,data))
        return ret


    def extract_feature(self,data):
        #F0 index range
        if self.feature == "pitch":
            return data[672:693]
        else:
            return data

    def get_label_name_list(self):
        if self.feature == "pitch":
            return self.label_list[672:693]
        else:
            return self.label_list

    def show_plot(self,dataset_list,datasetnames):
        # fg = seaborn.FacetGrid(data=df, hue='Gender', hue_order=_genders, aspect=1.61)
        # fg.map(plt.scatter, 'Weight (kg)', 'Height (cm)').add_legend()
        # seaborn.scatterplot(data=data[0], x='Height (cm)', y='Weight (kg)', hue='Gender')
        label_list = self.get_label_name_list()
        for i, label in enumerate(label_list):
            label = label[3:-2]
            label = label.split(' ')[1]
            # print(label)
            # label = label.split(' ')[2].split('_')[-1]
            for dataset,datasetname in zip(dataset_list,datasetnames):
                plt.scatter([datasetname for _ in dataset[i]], dataset[i], marker='o')
        # plt.scatter(self.get_label_name_list(),data)
            # plt.show()
            # plt.figure.Figure.set_size_inches(16,12)
            plt.title(label)
            plt.savefig(f"{label}.png")
            plt.clf()