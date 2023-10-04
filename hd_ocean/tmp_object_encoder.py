#%%
import os 
import warnings

import numpy as np
import pandas as pd

from copy import deepcopy

from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt

warnings.filterwarnings(action = 'ignore')


#%%
num_co = 25
num_po = 103
num_ship = 4
num_flag = 123

def transform(target):
    transformed = np.array(np.log(target + 1))
    return transformed


def inv_transform(pred):
    transformed = torch.exp(pred) - 1
    # transformed = np.exp(pred) - 1
    return transformed


class OceanDataset(Dataset):
    def __init__(self, path):
        self.path = path
    
        train_data = pd.read_csv(self.path)
        
        self.selected_col = [
            'ARI_CO', 'ARI_PO', 'SHIP_TYPE_CATEGORY', 'DIST', 'BREADTH', 'BUILT', 'DEADWEIGHT', 'DEPTH', 
            'DRAUGHT', 'GT', 'LENGTH', 'FLAG', 'ATA_LT', 'DUBAI', 'BRENT', 'WTI', 'BDI_ADJ', 'PORT_SIZE'
            ]

        feature = train_data[self.selected_col]
        try: 
            target = train_data.CI_HOUR
            # scaled_target = transform(target)
            self.scaled_target = torch.tensor(target.to_numpy()).to(torch.float32)
            # self.scaled_target = torch.tensor(scaled_target).to(torch.float32)
        except:
            scaled_target = torch.zeros(len(feature))
            self.scaled_target = torch.tensor(scaled_target).to(torch.float32)

        object_col = feature.select_dtypes(include = [object]).columns
        float_col = feature.select_dtypes(include = [float, int]).columns
        for col in object_col:
            col_unique = feature[col].unique()
            col_map = {col_unique[i]: int(i) for i in range(len(col_unique))}
            feature[col] = feature[col].replace(col_unique, col_map.values())

        feature.BREADTH[feature.BREADTH.isna()] = np.mean(feature.BREADTH)
        feature.DEPTH[feature.DEPTH.isna()] = np.mean(feature.DEPTH)
        feature.DRAUGHT[feature.DRAUGHT.isna()] = np.mean(feature.DRAUGHT)
        feature.LENGTH[feature.LENGTH.isna()] = np.mean(feature.LENGTH)
        
        float_feature = feature[float_col]
        object_feature = feature[object_col]
        
        self.float_feature = torch.tensor(float_feature.to_numpy()).to(torch.float32)
        self.object_feature = torch.tensor(object_feature.to_numpy()).to(torch.int32)
        
    def __len__(self):
        return len(self.scaled_target)
    
    def __col__(self):
        return self.selected_col
    
    def __getitem__(self, index):
        if type(index) == int:
            data = (self.float_feature[index], self.object_feature[index], self.scaled_target[index])
        else:
            data = [(self.float_feature[i], self.object_feature[i], self.scaled_target[i]) for i in index]
        return data


class MultiLayerPerceptron(nn.Module):
    def __init__(self, float_p, obj_p):
        super(MultiLayerPerceptron, self).__init__()        
        self.p1 = float_p
        self.p2 = obj_p
        
        self.emblayer1 = nn.Embedding(num_co, 300)
        self.emblayer2 = nn.Embedding(num_po, 300)
        self.emblayer3 = nn.Embedding(num_ship, 300)
        self.emblayer4 = nn.Embedding(num_flag, 300)
        
        torch.nn.init.xavier_uniform_(self.emblayer1.weight.data)
        torch.nn.init.xavier_uniform_(self.emblayer2.weight.data)
        torch.nn.init.xavier_uniform_(self.emblayer3.weight.data)
        torch.nn.init.xavier_uniform_(self.emblayer4.weight.data)
        
        self.layer1 = nn.Linear(self.p1, 300, bias = True)
        self.layer2 = nn.Linear(600, 300, bias = True)
        self.layer3 = nn.Linear(300, 100, bias = True)
        self.layer4 = nn.Linear(100, 1, bias = True)
        
        self.batchnorm1 = nn.BatchNorm1d(600)
        self.batchnorm2 = nn.BatchNorm1d(300)
        self.batchnorm3 = nn.BatchNorm1d(100)
        
    def forward(self, float_x, obj_x):
        float_embeddings = self.layer1(float_x)
        obj_embeddings = self.emblayer1(obj_x[:,0]) + self.emblayer2(obj_x[:,1]) + self.emblayer3(obj_x[:,2]) + self.emblayer4(obj_x[:,3])
        
        h1 = self.batchnorm1(torch.concat([float_embeddings, obj_embeddings], dim = 1))
        h2 = self.batchnorm2(nn.functional.tanh(self.layer2(h1)))
        h3 = self.batchnorm3(nn.functional.relu(self.layer3(h2)))
        output = nn.functional.relu(self.layer4(h3))
    
        return output


def train(model, device, loader, criterion, optimizer):
    model.train()
    
    for _, (float_x, obj_x, y) in enumerate(loader):
        float_x = float_x.to(torch.float32).to(device)
        obj_x = obj_x.to(device)
        y = y.to(torch.float32).to(device)
        
        pred = model(float_x, obj_x)
        
        optimizer.zero_grad()
        loss = criterion(pred, y.view(pred.shape))
        # loss = criterion(inv_transform(pred), inv_transform(y.view(pred.shape)))
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluation(model, device, loader, criterion):
    model.eval()
    
    loss_list = []
    
    for _, (float_x, obj_x, y) in enumerate(loader):
        float_x = float_x.to(torch.float32).to(device)
        obj_x = obj_x.to(device)
        y = y.to(torch.float32).to(device)
        
        pred = model(float_x, obj_x)
        loss = criterion(pred, y.view(pred.shape))
        # loss = criterion(inv_transform(pred), inv_transform(y.view(pred.shape)))
        loss_list.append(loss)
    
    return sum(loss_list)/len(loss_list)


#%%
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.mps.manual_seed(seed)

dataset = OceanDataset('data/train.csv')

num_train = round(len(dataset)*0.8)
num_test = len(dataset) - num_train

train_idx = np.random.choice(len(dataset), num_train, replace = False).astype(int)
val_idx = np.setdiff1d(range(len(dataset)), train_idx).astype(int)
train_dataset = dataset[train_idx]
val_dataset = dataset[val_idx]

batch_size = 256
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)


#%%
np.random.seed(seed)
torch.manual_seed(seed)
torch.mps.manual_seed(seed)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

num_float_feat = dataset[0][0].shape[-1]
num_obj_feat = dataset[0][1].shape[-1]

model = MultiLayerPerceptron(num_float_feat, num_obj_feat)
model = model.to(device)

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

epochs = 500

best_val = 100
for epoch in range(1, epochs+1):
    train(model, device, train_loader, criterion, optimizer)
    
    train_loss = evaluation(model, device, train_loader, criterion)
    val_loss = evaluation(model, device, val_loader, criterion)
    
    if val_loss < best_val:
        best_val = val_loss
        best_model_param = model.state_dict()
    
    # if epoch % 10 == 0:
    print(f'epoch : {epoch}')
    print(f'train loss = {train_loss:.3f}, val loss = {val_loss:.3f}')

torch.save(best_model_param, 'best_model.pth')
torch.save(model.state_dict(), 'final_model.pth')


# %%
test_dataset = OceanDataset('data/test.csv')
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

model = MultiLayerPerceptron(num_float_feat, num_obj_feat).to(device)
model.load_state_dict(best_model_param)
model.eval()


pred_list = []
for _, (float_x, obj_x, y) in enumerate(test_loader):
    float_x = float_x.to(torch.float32).to(device)
    obj_x = obj_x.to(device)
    y = y.to(torch.float32).to(device)
    
    pred = model(float_x, obj_x)
    pred_list.append(pred.view(-1))

test_pred = torch.cat(pred_list).detach().cpu().numpy()


# %%
submission = pd.read_csv('data/sample_submission.csv')
submission.CI_HOUR = test_pred
submission.to_csv('pred.csv', header = True, index = False)
