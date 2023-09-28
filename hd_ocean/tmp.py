#%%
import os 
import warnings

import numpy as np
import pandas as pd

from copy import deepcopy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import matplotlib.pyplot as plt

warnings.filterwarnings(action = 'ignore')


#%%
os.listdir('data')

train_data = pd.read_csv('data/train.csv', header = 0)
train_data.columns

raw_target = train_data.CI_HOUR

plt.hist(raw_target)
plt.hist(np.log(raw_target+1))

train_data.dtypes.unique()

# float column
float_col = train_data.select_dtypes(include = [float]).columns

for col in float_col:
    plt.hist(train_data[col])
    plt.title(col)
    plt.show()
    plt.close()

# int column
train_data.dtypes
train_data.DEADWEIGHT
train_data.GT
train_data.ATA_LT

# object column
object_col = train_data.select_dtypes(include = [object]).columns

train_data.SHIP_TYPE_CATEGORY.value_counts()
train_data.ID.value_counts()
train_data.SHIPMANAGER.value_counts()
train_data.FLAG.value_counts()


co_unique = train_data.ARI_CO.unique()
len(co_unique)
train_data.ARI_CO.value_counts()

fig, axs = plt.subplots(len(co_unique), figsize = (5, 25))
for i in range(len(co_unique)):
    axs[i].hist(np.log(train_data.CI_HOUR[train_data.ARI_CO == co_unique[i]] + 1))
    axs[i].set_title(co_unique[i])
plt.show()
plt.close()


po_unique = train_data.ARI_PO.unique()
len(po_unique)
train_data.ARI_PO.value_counts()

fig, axs = plt.subplots(len(po_unique), figsize = (5, 35))
for i in range(len(po_unique)):
    axs[i].hist(np.log(train_data.CI_HOUR[train_data.ARI_PO == po_unique[i]] + 1))
    axs[i].set_title(po_unique[i])
plt.show()
plt.close()


# plt.hist(train_data.DIST)

# fig, axs = plt.subplots(len(co_unique), figsize = (5, 25))
# for i in range(len(co_unique)):
#     axs[i].hist(train_data.DIST[train_data.ARI_CO == co_unique[i]])
#     axs[i].set_title(co_unique[i])
# plt.show()
# plt.close()



#%%
''' 
    ATA: 실제 정박 시각
    ID: 선박 식별 일련번호
    DEADWEIGHT, GT: scaling 필요할듯?
    SHIPMANAGER: 선박 소유주
    
    ATA, ID, SHIPMANAGER 제외
'''

def transform(target):
    transformed = np.array(np.log(target + 1))
    return transformed

def inv_transform(pred):
    transformed = torch.exp(pred) - 1
    # transformed = np.exp(pred) - 1
    return transformed

selected_col = [
    'ARI_CO', 'ARI_PO', 'SHIP_TYPE_CATEGORY', 'DIST', 'BREADTH', 'BUILT', 'DEADWEIGHT', 'DEPTH', 
    'DRAUGHT', 'GT', 'LENGTH', 'FLAG', 'ATA_LT', 'DUBAI', 'BRENT', 'WTI', 'BDI_ADJ', 'PORT_SIZE'
    ]
feature_tmp = train_data[selected_col]
feature = deepcopy(feature_tmp)

target = train_data.CI_HOUR
scaled_target = transform(raw_target)

object_col = feature.select_dtypes(include = [object]).columns
for col in object_col:
    col_unique = feature[col].unique()
    col_map = {col_unique[i]: int(i) for i in range(len(col_unique))}
    feature[col] = feature[col].replace(col_unique, col_map.values())

na_col = feature.columns[feature.isna().sum() != 0]
feature[na_col].dtypes

feature.BREADTH[feature.BREADTH.isna()] = np.mean(feature.BREADTH)
feature.DEPTH[feature.DEPTH.isna()] = np.mean(feature.DEPTH)
feature.DRAUGHT[feature.DRAUGHT.isna()] = np.mean(feature.DRAUGHT)
feature.LENGTH[feature.LENGTH.isna()] = np.mean(feature.LENGTH)

feature = np.array(feature)
x_train, x_test, y_train, y_test = train_test_split(feature, scaled_target, test_size = 0.2)


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

model = RandomForestRegressor()
model.fit(x_train, y_train)
pred = model.predict(x_test)
inv_trans_pred = inv_transform(pred)

mean_absolute_error(y_test, pred)
mean_absolute_error(inv_transform(y_test), inv_trans_pred)

model.feature_importances_

plt.figure(figsize = (20, 10))
plt.bar(feature_tmp.columns, model.feature_importances_)
plt.show()
plt.close()


#%%
test_data = pd.read_csv('data/test.csv')
submission = pd.read_csv('data/sample_submission.csv')

test_df = test_data[selected_col]
for col in object_col:
    col_unique = test_df[col].unique()
    col_map = {col_unique[i]: int(i) for i in range(len(col_unique))}
    test_df[col] = test_df[col].replace(col_unique, col_map.values())

test_pred = model.predict(test_df)
submission.CI_HOUR = test_pred

submission.to_csv('tmp_pred.csv', header = True, index = False)


#%%
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

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
            scaled_target = transform(target)
        except:
            scaled_target = torch.zeros(len(feature))

        object_col = feature.select_dtypes(include = [object]).columns
        for col in object_col:
            col_unique = feature[col].unique()
            col_map = {col_unique[i]: int(i) for i in range(len(col_unique))}
            feature[col] = feature[col].replace(col_unique, col_map.values())

        feature.BREADTH[feature.BREADTH.isna()] = np.mean(feature.BREADTH)
        feature.DEPTH[feature.DEPTH.isna()] = np.mean(feature.DEPTH)
        feature.DRAUGHT[feature.DRAUGHT.isna()] = np.mean(feature.DRAUGHT)
        feature.LENGTH[feature.LENGTH.isna()] = np.mean(feature.LENGTH)
        
        self.feature = torch.tensor(feature.to_numpy())
        self.scaled_target = torch.tensor(scaled_target)
        
    def __len__(self):
        return len(self.scaled_target)
    
    def __col__(self):
        return self.selected_col
    
    def __getitem__(self, index):
        if type(index) == int:
            data = (self.feature[index], self.scaled_target[index])
        else:
            data = [(self.feature[i], self.scaled_target[i]) for i in index]
        return data


class MultiLayerPerceptron(nn.Module):
    def __init__(self, p):
        super(MultiLayerPerceptron, self).__init__()        
        self.p = p
        # self.layer1 = nn.Linear(p, 300, bias = True)
        # self.layer2 = nn.Linear(300, 600, bias = True)
        # self.layer3 = nn.Linear(600, 300, bias = True)
        # self.layer4 = nn.Linear(300, 100, bias = True)
        # self.layer5 = nn.Linear(100, 1, bias = True)
        
        # self.batchnorm1 = nn.BatchNorm1d(300)
        # self.batchnorm2 = nn.BatchNorm1d(600)
        # self.batchnorm3 = nn.BatchNorm1d(300)
        # self.batchnorm4 = nn.BatchNorm1d(100)
        
        self.layer1 = nn.Linear(p, 100000)
        self.layer2 = nn.Linear(100000, 1)
        
        self.batchnorm1 = nn.BatchNorm1d(100000)
        
    def forward(self, x):
        # h1 = self.batchnorm1(self.layer1(x))
        # h2 = self.batchnorm2(nn.functional.relu(self.layer2(h1)))
        # h3 = self.batchnorm3(nn.functional.tanh(self.layer3(h2)))
        # h4 = self.batchnorm4(nn.functional.relu(self.layer4(h3)))
        # output = nn.functional.elu(self.layer5(h4))

        h1 = self.batchnorm1(nn.functional.relu(self.layer1(x)))
        output = nn.functional.relu(self.layer2(h1))
    
        return output


class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNetwork, self).__init__()        
        
        self.conv1 = nn.Conv1d(in_channels = 1, out_channels = 4, kernel_size = 1, groups = 1)
        
        self.conv2 = nn.Conv1d(in_channels = 4, out_channels = 32, kernel_size = 2)
        self.pool1 = nn.MaxPool1d(2)
        self.batchnorm1 = nn.BatchNorm1d(32)
        
        self.conv3 = nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 2)
        self.pool2 = nn.AvgPool1d(2)
        self.batchnorm2 = nn.BatchNorm1d(64)
        
        self.flat = nn.Flatten()
        
        self.lin1 = nn.Linear(64 * 3, 50)
        self.batchnorm3 = nn.BatchNorm1d(50)

        self.lin2 = nn.Linear(50, 1)

    def forward(self, x):
        x = x.view(x.shape[0], 1, x.shape[1])
        
        h1 = self.conv1(x)
        h1 = nn.functional.relu(h1)
        
        h2 = self.batchnorm1(self.pool1(self.conv2(h1)))
        h2 = nn.functional.relu(h2)
        
        h3 = self.batchnorm2(self.pool2(self.conv3(h2)))
        h3 = nn.functional.relu(h3)
        h3 = self.flat(h3)
        
        h4 = self.batchnorm3(self.lin1(h3))
        h4 = nn.functional.relu(h4)
        
        output = self.lin2(h4)
    
        return output


def train(model, device, loader, criterion, optimizer):
    model.train()
    
    for _, (x, y) in enumerate(loader):
        x = x.to(torch.float32).to(device)
        y = y.to(torch.float32).to(device)
        
        pred = model(x)
        
        optimizer.zero_grad()
        # loss = criterion(pred, y.view(pred.shape))
        loss = criterion(inv_transform(pred), inv_transform(y.view(pred.shape)))
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluation(model, device, loader, criterion):
    model.eval()
    
    loss_list = []
    
    for _, (x, y) in enumerate(loader):
        x = x.to(torch.float32).to(device)
        y = y.to(torch.float32).to(device)
        
        pred = model(x)
        # loss = criterion(pred, y.view(pred.shape))
        loss = criterion(inv_transform(pred), inv_transform(y.view(pred.shape)))
        loss_list.append(loss)
    
    return sum(loss_list)/len(loss_list)


def transform(target):
    transformed = np.array(np.log(target + 1))
    return transformed


def inv_transform(pred):
    transformed = torch.exp(pred) - 1
    # transformed = np.exp(pred) - 1
    return transformed


#%%
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.mps.manual_seed(seed)

dataset = OceanDataset('data/train.csv')

num_feat = dataset[0][0].shape[-1]
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

model = ConvolutionalNetwork()
# model = MultiLayerPerceptron(num_feat)
model = model.to(device)

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

epochs = 100

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

torch.save(best_model_param, 'cnn_best_model.pth')
torch.save(model.state_dict(), 'cnn_final_model.pth')


#%%
np.random.seed(seed)
torch.manual_seed(seed)
torch.mps.manual_seed(seed)

model = ConvolutionalNetwork()
model.load_state_dict(torch.load('cnn_best_model.pth'))
# model = MultiLayerPerceptron(num_feat)
# model.load_state_dict(torch.load('best_model.pth'))

# device = torch.device('cpu')
model = model.to(device)
model.eval()


loss_list = []
for _, (x, y) in enumerate(val_loader):
    x = x.to(torch.float32).to(device)
    y = y.to(torch.float32).to(device)
    
    pred = model(x)
    loss = criterion(inv_transform(pred), inv_transform(y.view(pred.shape)))
    loss_list.append(loss)

sum(loss_list)/len(loss_list)


#%%
np.random.seed(seed)
torch.manual_seed(seed)
torch.mps.manual_seed(seed)

model = ConvolutionalNetwork()
model.load_state_dict(torch.load('cnn_best_model.pth'))
# model = MultiLayerPerceptron(num_feat)
# model.load_state_dict(torch.load('best_model.pth'))
model = model.to(device)
model.eval()

test_data = pd.read_csv('data/test.csv')
submission = pd.read_csv('data/sample_submission.csv')

test_dataset = OceanDataset('data/test.csv')
test_loader = DataLoader(test_dataset, 512, shuffle = False)

pred_list = []
for _, (x, y) in enumerate(test_loader):
    x = x.to(torch.float32).to(device)
    y = y.to(torch.float32).to(device)
    
    pred = model(x)
    pred_list.append(pred.view(-1))

test_pred = torch.cat(pred_list).detach().cpu().numpy()

submission.CI_HOUR = test_pred
submission.to_csv('cnn_pred.csv', header = True, index = False)
