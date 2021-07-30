import pickle
from functions import normalization_ops, wLoss
from data_loader import load_emotion_corpus_WC
from setting import ROOT_PATH
import torch
import os 
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import numpy as np 
import matplotlib.pyplot as plt
import fnmatch
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
        return data['x_data'],data['y_data']

# FGSM 공격 코드
def fgsm_attack(image, epsilon, data_grad,clip_range):
    # data_grad 의 요소별 부호 값을 얻어옵니다
    sign_data_grad = -1 * data_grad.sign()
    # 입력 이미지의 각 픽셀에 sign_data_grad 를 적용해 작은 변화가 적용된 이미지를 생성합니다
    perturbed_image = image + epsilon*sign_data_grad

    # 값 범위를 [0,1]로 유지하기 위해 자르기(clipping)를 추가합니다
    # perturbed_image = torch.clamp(perturbed_image, 0, 1)

    # 작은 변화가 적용된 이미지를 리턴합니다
    return perturbed_image

def show_plot():
    pass
def save_result():
    pass

def get_pth_files(path):
    return fnmatch.filter(os.listdir(path),'*.pth')

def get_output(x_data,y_data,model,loss_func):
    x_data.requires_grad = True
    class_output, _, _ = model(input_data=x_data, alpha=0)

    pred = class_output.data.max(1, keepdim=True)[1]
    loss = loss_func(class_output, pred, y_data)
    return pred,loss

def test(x_data,y_data,model,loss_func):
    x_data = torch.Tensor([x_data]).to(device).cuda()
    y_data = torch.Tensor([y_data]).to(device).long().cuda()
    prediction, loss = get_output(x_data,y_data,model,loss_func)
    
    if prediction.item() != y_data.item():
        return y_data

    model.zero_grad()
    loss.backward()
    data_grad = x_data.grad.data
    


if __name__ == '__main__':
    x_dataset,y_dataset = load_data()
    for ep in range(30):
        pre_trained_models = get_pth_files(MODEL_PATH)
        for model_path in pre_trained_models:
            loss_func = wLoss().cuda()
            model = torch.load(os.path.join(MODEL_PATH,model_path))
            model.eval()
            for x_data,y_data in zip(x_dataset,y_dataset):
                output, gradient, features = test(x_data,y_data,model,loss_func)

# x_train, y_train, x_valid, y_valid, x_test, y_test, ys_test = load_emotion_corpus_WC("CREMA-D", DATA_PATH,0)
x_valid, y_valid, _, _, x_train, y_train, ys_test = load_emotion_corpus_WC("CREMA-D", DATA_PATH,0)

feat_mu = np.mean(x_train,axis=0)
feat_st = np.std(x_train, axis=0)

x_train  = normalization_ops(feat_mu, feat_st, x_train)
# x_valid  = normalization_ops(feat_mu, feat_st, x_valid)
# x_test   = normalization_ops(feat_mu, feat_st, x_test)

loss_func   = wLoss().cuda()

max_ep = 0
max_diff = 0

result_acc_plt_map = []

for ep in range(30):
    ep_val = (ep)/100
    result_acc_diff = [0,0,0]
    print(f"epsilon: {ep_val}")
    for i in range(5):
        model = torch.load(os.path.join(MODEL_PATH,f"WC_fold0_seed{i}.pth"))
        model.eval()
        eval_wa = []
        eval_ua = []

        eval_wa2 = []
        eval_ua2 = []
        eval_ua3 = []
        test = np.zeros(len(x_train[0]))
        for i in range(len(x_train)):
            # x_train_batch = torch.Tensor(self.dataset[DataType.X_TRAIN][samples]).to(self.setting.device).cuda()
            # y_train_batch = torch.Tensor(self.dataset[DataType.Y_TRAIN][samples]).to(self.setting.device).long().cuda()
            x_eval = torch.Tensor([x_train[i]]).to(device).cuda()
            y_eval = torch.Tensor([y_train[i]]).to(device).long().cuda()

            x_eval.requires_grad = True
            class_output, _, _ = model(input_data=x_eval, alpha=0)
            pred = class_output.data.max(1, keepdim=True)[1]

            if pred.item() != y_eval.item():
                # eval_wa.append(0)
                eval_ua.append(0)
                # eval_wa2.append(0)
                eval_ua2.append(0)
                eval_ua3.append(0)
                continue

            loss = loss_func(class_output, pred, y_eval)
            model.zero_grad()
            loss.backward()
            data_grad = x_eval.grad.data

            test = test + np.array(data_grad.data[0].cpu())

            perturbed_data = fgsm_attack(x_eval, ep_val, data_grad)
            class_output2,_,_ = model(perturbed_data,alpha=0)
            pred2 = class_output2.data.max(1, keepdim=True)[1]
            # print(y_eval,pred,pred2)

            perturbed_data_clip = fgsm_attack_with_clip(x_eval, ep_val, data_grad)
            class_output3,_,_ = model(perturbed_data_clip,alpha=0)
            pred3 = class_output3.data.max(1, keepdim=True)[1]


            # eval_wa.append(0 if pred.item() != y_eval.item() else 1)
            eval_ua.append(0 if pred.item() != y_eval.item() else 1)
            # eval_wa2.append(0 if pred2.item() != y_eval.item() else 1)
            eval_ua2.append(0 if pred2.item() != y_eval.item() else 1)
            eval_ua3.append(0 if pred3.item() != y_eval.item() else 1)

            del x_eval, y_eval
        # print(f"{np.mean(eval_wa)*100:.2f},{np.mean(eval_wa2)*100:.2f}%")
        result_acc_diff[0] += np.mean(eval_ua)*100
        result_acc_diff[1] += np.mean(eval_ua2)*100
        result_acc_diff[2] += np.mean(eval_ua3)*100

    result_acc_diff  = [iter/5 for iter in result_acc_diff]
    result_acc_plt_map.append(result_acc_diff)
    print(f"{result_acc_diff[0]:.2f}%, {result_acc_diff[1]:.2f}%")
    if max_diff < result_acc_diff[0] - result_acc_diff[1]:
        max_diff = result_acc_diff[0] - result_acc_diff[1]
        max_ep = ep_val
        # plt.plot(np.absolute(test)) # plotting by columns
        # plt.show()

x_position = np.array([i for i in range(len(result_acc_plt_map))])
# a = np.array(result_acc_plt_map)[:,0]
b = np.array(result_acc_plt_map)[:,1]
c = np.array(result_acc_plt_map)[:,2]
# plt.bar(x_position-0.2,a, width = 0.2)
plt_1 = plt.bar(x_position-0.1,b, width = 0.2)
plt_2 = plt.bar(x_position+0.1,c, width = 0.2)
plt.legend((plt_1,plt_2), ('without clip','with clip'), fontsize=15)
# plt.xticks(range(min(x1+x2), max(x1+x2)+1)) 
plt.show()
print(ep_val,max_ep)
    
