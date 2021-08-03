import csv
import pickle
from functions import normalization_ops, wLoss
from setting import ROOT_PATH
import torch
import os 
import numpy as np 
import matplotlib.pyplot as plt
import fnmatch
import matplotlib.patches as mpatches
from tqdm import tqdm
import json

'''
# 1 엡실론 더한 값의 feature랑 기존 feature를 비교하는 값을 뽑는다
# 2 gradient가 큰 값만 변형했을 때 정확도 차이를 본다
# 3 fgsm의 결과값을 본다
# 4 fgsm에서 sign이 아닌 direction을 그대로 반영하는 결과값도 본다
# 5 위 4개를 전부 데이터화 시켜서 저장한다
'''

DATA_PATH = os.path.join(ROOT_PATH,"CREMA-D","emobase2010")
MODEL_PATH = os.path.join(ROOT_PATH,"CREMA-D","emobase2010","WC0716_JY")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data():
    train_filename = f"{DATA_PATH}.pickle"
    
    with open(train_filename, 'rb') as handle:
        data = pickle.load(handle)
        x_train = data['x_data']
        y_train = data['y_data']
        feat_mu = np.mean(x_train,axis=0)
        feat_st = np.std(x_train, axis=0)

        x_train  = normalization_ops(feat_mu, feat_st, x_train)
        return x_train, y_train

# FGSM 공격 코드
def fgsm_attack(data, epsilon, data_grad,clip_range=None):
    # data_grad 의 요소별 부호 값을 얻어옵니다
    sign_data_grad = -1 * data_grad.sign()
    # 입력 이미지의 각 픽셀에 sign_data_grad 를 적용해 작은 변화가 적용된 이미지를 생성합니다
    perturbed_data = data + epsilon*sign_data_grad

    # 값 범위를 [0,1]로 유지하기 위해 자르기(clipping)를 추가합니다
    if clip_range is not None:
        perturbed_data = torch.clamp(perturbed_data, clip_range[0], clip_range[1])

    # 작은 변화가 적용된 이미지를 리턴합니다
    return perturbed_data.to(device).cuda()

def feature_attack(data,epsilon,data_grad,feature_idx):
    # data_grad 의 요소별 부호 값을 얻어옵니다
    sign_data_grad = -1 * data_grad.sign()
    # 입력 이미지의 각 픽셀에 sign_data_grad 를 적용해 작은 변화가 적용된 이미지를 생성합니다
    perturbed_data = data.clone()
    perturbed_data[0][feature_idx] += epsilon*sign_data_grad[0][feature_idx]

    del sign_data_grad
    # 작은 변화가 적용된 이미지를 리턴합니다
    return perturbed_data

def show_plot():
    pass
def save_result():
    pass

def get_pth_files(path):
    return fnmatch.filter(os.listdir(path),'*.pth')

def get_output(x_data,y_data,model,loss_func,requires_grad=False):
    if requires_grad == True:
        x_data.requires_grad = requires_grad
    class_output, _, _ = model(input_data=x_data, alpha=0)

    pred = class_output.data.max(1, keepdim=True)[1]
    loss = loss_func(class_output, pred, y_data)

    return pred,loss

def test(x_data,y_data,model,loss_func,epsilon,feature_names):
    x_data = torch.Tensor([x_data]).to(device).cuda()
    y_data = torch.Tensor([y_data]).to(device).long().cuda()
    prediction, loss = get_output(x_data,y_data,model,loss_func,requires_grad=True)
    
    if prediction.item() != y_data.item():
        return {}

    model.zero_grad()
    loss.backward()
    data_grad = x_data.grad.data

    attack_data = { "original": x_data,
                    # "with_clip":fgsm_attack(x_data,epsilon,data_grad,[-10,10]),
                    "without_clip":fgsm_attack(x_data,epsilon,data_grad)}
    for i in range(len(feature_names)):
        attack_data[f"feature_{feature_names[i]}"] = feature_attack(x_data,epsilon,data_grad,i)

    attack_result = {}
    for key,data in attack_data.items():
        attack_result[key] = get_output(data,y_data,model,loss_func)
    return_data = {key:(attack_data[key],attack_result[key][0],attack_result[key][1]) 
                        for key in attack_data.keys()}

    return return_data

def load_smile_feature_names():
    path = os.path.join(DATA_PATH,"opensmile_props.csv")
    csvfile = open(path,'r')
    reader = [each for each in csv.reader(csvfile, delimiter=';')]
    csvfile.close()
    feats_name = [each[0].split(' ')[1] for each in reader[3:-5]]
    return feats_name

def show_feature_plot(features_model):
    feature_info = {}
    for key,feature_list in features_model.items():
        feature_info[key] = {}
        for feature_name,feature_values in feature_list.items():
            feature_values = np.array(feature_values)
            feature_info[key][feature_name] = {
                "min":np.min(feature_values),
                "max":np.max(feature_values),
                "mean":np.mean(feature_values)}

    #이 seed에 대해서 feature 범위 보여주기
    colors = ('b','r','green')
    for i,(key,feature_list) in enumerate(feature_info.items()):
        for j,(feature_name,feature_values) in enumerate(feature_list.items()):
            plt.vlines(x=j+i*0.2, ymin=feature_values['min'], ymax=feature_values['max'], color=colors[i], label=key)
            if j > 10:
                break
    
    legends = []
    for color,label in zip(colors,feature_info.keys()):
        legends.append(mpatches.Patch(color=color, label=label))

    plt.legend(handles=legends)
    plt.show()

if __name__ == '__main__':
    feature_names = load_smile_feature_names()
    x_dataset,y_dataset = load_data()
    loss_func = wLoss().cuda()
    for i in range(30):
        epsilon = i/100 + 0.1
        pre_trained_models = get_pth_files(MODEL_PATH)

        ua_total = []
        features_total = []

        for model_path in pre_trained_models:
            ua_model = {}
            features_model = {}
            model = torch.load(os.path.join(MODEL_PATH,model_path))
            model.eval()
            for x_data,y_data in tqdm(zip(x_dataset,y_dataset),total=len(x_dataset)):
                result = test(x_data,y_data,model,loss_func,epsilon,feature_names)
                for key,(features,prediction,loss) in result.items():
                    if key not in ua_model.keys():
                        ua_model[key] = 0
                    ua_model[key] += 1 if y_data == prediction.item() else 0

                    if key not in features_model.keys():
                        features_model[key] = {feature:[] for feature in feature_names}

                    # features = features[0].cpu().detach().numpy()
                    # print(features)
                    # for feature, value in zip(feature_names,features):
                    #     features_model[key][feature].append(value)

            ua_model = {key:value/len(x_dataset)*100 for key,value in ua_model.items()}

            ua_list = list(ua_model.items())
            ua_list.sort(key=lambda value: value[1])
            # ua_list = ua_list[:20]
            ua_list_diff = [(key,ua_model['original'] - elem) for key,elem in ua_list]

            ua_top_dict = {key:value for key,value in ua_list_diff}
            # print(ua_top_dict)
            with open(f'./{model_path.split(".")[0]}_{epsilon}.json','w') as f:
                json.dump(ua_top_dict,f)

            # show_feature_plot(features_model)

