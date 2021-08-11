from utils.setting import ROOT_PATH
import torch
import os 
from tqdm import tqdm
import torch.nn as nn

'''
# 1 엡실론 더한 값의 feature랑 기존 feature를 비교하는 값을 뽑는다
# 2 gradient가 큰 값만 변형했을 때 정확도 차이를 본다
# 3 fgsm의 결과값을 본다
# 4 fgsm에서 sign이 아닌 direction을 그대로 반영하는 결과값도 본다
# 5 위 4개를 전부 데이터화 시켜서 저장한다
'''


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Tester():
    def __init__(self,model:nn.Module,loss_func:nn.Module):
        self.model = model
        self.loss_func = loss_func
    
    def test(self,x_dataset:torch.Tensor,y_dataset:torch.Tensor) -> float:
        self.model.eval()
        sum = 0
        for x_data,y_data in tqdm(zip(x_dataset,y_dataset),total=len(x_dataset)):
            x_data = x_data.unsqueeze(0)
            y_data = y_data.unsqueeze(0)
            pred,loss = self.get_output(x_data,y_data)
            sum += 1 if pred.item() == int(y_data.item()) else 0
        return sum/len(x_dataset)*100

    def get_gradient(self,x_data:torch.Tensor,y_data:torch.Tensor) -> torch.Tensor:
        self.model.eval()
        prediction, loss = self.get_output(x_data,y_data,requires_grad=True)
        if prediction.item() != y_data.item():
            return {}

        self.model.zero_grad()
        loss.backward()

        return x_data.grad.data

    def get_output(self,x_data:torch.Tensor,y_data:torch.Tensor,requires_grad=False) -> 'tuple[torch.Tensor,torch.Tensor]':
        if requires_grad == True:
            x_data.requires_grad = requires_grad
        
        class_output, _, _ = self.model(input_data=x_data, alpha=0)
        class_output:torch.Tensor = class_output

        pred = class_output.data.max(1, keepdim=True)[1]
        loss:torch.Tensor = self.loss_func(class_output, pred, y_data)

        return pred,loss