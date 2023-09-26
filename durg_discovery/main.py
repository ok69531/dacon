#%%
import numpy as np
import pandas as pd
from copy import deepcopy

import torch
from torch import nn
from torch import optim

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from module.loss import RMSELoss
from module.mol import smiles2graph
from module.model import GNNGraphPred
from module.dataset_drug import DrugDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


#%%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data = pd.read_csv('data/train/raw/train.csv')
addit_feature_columns = ['AlogP', 'Molecular_Weight', 'Num_H_Acceptors', 
                      'Num_H_Donors', 'Num_RotatableBonds', 'LogD',
                      'Molecular_PolarSurfaceArea']
# scaler = MinMaxScaler()
# addit_feature = scaler.fit_transform(data[addit_feature_columns])
addit_feature = torch.from_numpy(data[addit_feature_columns].to_numpy()).to(torch.float32).to(device)


dataset = DrugDataset('data/train')
test_dataset = DrugDataset('data/test')

train_idx, val_idx = train_test_split(np.arange(len(data)), train_size = 0.9)

train_loader = DataLoader(dataset[train_idx], batch_size = 32, shuffle = True, num_workers = 0)
val_loader = DataLoader(dataset[val_idx], batch_size = 32, shuffle = False, num_workers = 0)
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False, num_workers = 0)


#%%
def train(model, device, loader, criterion, optimizer):
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            loss = criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()


# def aa_train(model, device, loader, criterion, optimizer):
#     model.train()
    
#     m = 3
#     max_pert = 0.001
#     step_size = 0.001
    
#     for step, batch in enumerate(loader):
#         batch = batch.to(device)
        
#         if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
#             pass
#         else:
#             graph_embedding = model.pool(model.gnn(batch), batch.batch)
#             perturb = torch.FloatTensor(graph_embedding.shape[0], graph_embedding.shape[1]).uniform_(-max_pert, max_pert).to(device)
#             perturb.requires_grad_()
            
#             pred = model.graph_pred_linear(graph_embedding + perturb)
            
#             is_labeled = batch.y == batch.y
            
#             optimizer.zero_grad()
#             loss = criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
#             loss /= m
            
#             for _ in range(m-1):
#                 loss.backward()
#                 perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
#                 perturb.data = perturb_data.data
#                 perturb.grad[:] = 0
                
#                 tmp_graph_embedding = model.pool(model.gnn(batch), batch.batch) + perturb
#                 tmp_pred = model.graph_pred_linear(tmp_graph_embedding)
                
#                 loss = 0
#                 loss = criterion(tmp_pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
#                 loss /= m
                        
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()


@torch.no_grad()
def evaluation(model, device, loader, criterion):
    model.eval()
    
    # y_true = []
    loss_list = []
    
    for _, batch in enumerate(loader):
        batch = batch.to(device)
        
        if batch.x.shape[0] == 1:
            pass
        else:
            pred = model(batch)
            
            y = batch.y.view(pred.shape).to(torch.float32)
            is_labeled = y == y
            # y_true.append(y[is_labeled].detach().cpu())
            
            loss = criterion(pred[is_labeled], y[is_labeled])
            loss_list.append(loss)
        
    # y_true = torch.cat(y_true, dim = 0).numpy()
    loss_list = torch.stack(loss_list)
    
    return sum(loss_list)/len(loss_list)


#%%
seed = 42
torch.manual_seed(seed)

num_task = 2
num_layer = 5
emb_dim = 300
gnn_type = 'gin'
graph_pooling = 'mean'
dropout_ratio = 0.3
JK = 'last'
virtual = False
residual = False 
lr = 0.001

gnn_model = GNNGraphPred(num_tasks = num_task, num_layer = num_layer, 
                         emb_dim = emb_dim, gnn_type = gnn_type,
                         graph_pooling = graph_pooling, drop_ratio = dropout_ratio, JK = JK, 
                         virtual_node = virtual, residual = residual)
gnn_model = gnn_model.to(device)

optimizer = optim.Adam(gnn_model.parameters(), lr = lr)
criterion = RMSELoss()


best_epoch = 0
best_val_loss = 100
for epoch in range(1, 100):
    print(f'=== epoch {epoch}')
    
    train(gnn_model, device, train_loader, criterion, optimizer)
    # aa_train(gnn_model, device, train_loader, criterion, optimizer)
    
    train_loss = evaluation(gnn_model, device, train_loader, criterion)
    val_loss = evaluation(gnn_model, device, val_loader, criterion)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        
        gnn_model_params = gnn_model.state_dict()
        
    print(f'train loss: {train_loss:.4f}, val loss: {val_loss:.4f}')

gnn_check_points = {'epoch': best_epoch,
                    'gnn_model_state_dict': gnn_model_params,
                    # 'path': path,
                    'optimizer_state_dict': optimizer.state_dict()}


#%%
class FullyConnectedNetwork(nn.Module):
    def __init__(self, p, num_task):
        super(FullyConnectedNetwork, self).__init__()
        self.p = p
        self.num_task = num_task 
        
        self.lin1 = nn.Linear(p, 100)
        self.lin2 = nn.Linear(100, 30)
        self.lin3 = nn.Linear(30, num_task)
    
    def forward(self, x):
        h = nn.functional.relu(self.lin1(x))
        h = nn.functional.tanh(self.lin2(h))
        prob = nn.functional.sigmoid(self.lin3(h))
        
        return prob


def reg_train(model, feat_extract_model, device, loader, addit_feat, criterion, optimizer):
    model.train()
    
    for _, batch in enumerate(loader):
        graph_feat = feat_extract_model.pool(feat_extract_model.gnn(batch), batch.batch)
        batch_addit_feat = addit_feat[batch.id.numpy()]
        feat = torch.cat([graph_feat, batch_addit_feat], axis = 1)
        
        pred = model(feat)
        y = batch.y
        is_labeled = y == y
    
        optimizer.zero_grad()
        loss = criterion(pred[is_labeled], y[is_labeled])
        loss.backward()
        optimizer.step()


@torch.no_grad()
def reg_eval(model, feat_extract_model, device, loader, addit_feat, criterion):
    model.eval()
    
    loss_list = []
    
    for _, batch in enumerate(loader):
        graph_feat = feat_extract_model.pool(feat_extract_model.gnn(batch), batch.batch)
        batch_addit_feat = addit_feat[batch.id.numpy()]
        feat = torch.cat([graph_feat, batch_addit_feat], axis = 1)
        
        pred = model(feat)
        y = batch.y
        is_labeled = y == y
    
        loss = criterion(pred[is_labeled], y[is_labeled])
    
        loss_list.append(loss)
    
    loss_list = torch.stack(loss_list)
    
    return sum(loss_list)/len(loss_list)


#%%
feat_extract_model = deepcopy(gnn_model)
feat_extract_model.load_state_dict(gnn_check_points['gnn_model_state_dict'])
feat_extract_model.eval()

num_addit_feat = addit_feature.shape[1]

reg_model = FullyConnectedNetwork(emb_dim + num_addit_feat, num_task)
optimizer = optim.SGD(reg_model.parameters(), lr = 0.001)

best_epoch = 0
best_val_loss = 100
for epoch in range(1, 100):
    print(f'=== epoch {epoch}')
    
    reg_train(reg_model, feat_extract_model, device, train_loader, addit_feature, criterion, optimizer)
    
    train_loss = reg_eval(reg_model, feat_extract_model, device, train_loader, addit_feature, criterion)
    val_loss = reg_eval(reg_model, feat_extract_model, device, val_loader, addit_feature, criterion)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        
        reg_model_params = reg_model.state_dict()
        
    print(f'train loss: {train_loss:.4f}, val loss: {val_loss:.4f}')

reg_check_points = {'epoch': best_epoch,
                    'gnn_model_state_dict': reg_model_params,
                    # 'path': path,
                    'optimizer_state_dict': optimizer.state_dict()}


#%%
test_data = pd.read_csv('data/test/raw/test.csv')
submission = pd.read_csv('data/sample_submission.csv')

test_addit_feature = test_data[addit_feature_columns].to_numpy()

pred_model = deepcopy(gnn_model)
pred_model.load_state_dict(gnn_check_points['gnn_model_state_dict'])
pred_model.eval()

pred_list = []
for step, batch in enumerate(test_loader):
    # embedding_tmp = gnn_model.(pool(model(batch), batch.btach))
    # batch_addit_feature = torch.from_numpy(test_addit_feature[batch.id.numpy()]).to(torch.float32).to(device)
    # pred_tmp = torch.cat([embedding_tmp, batch_addit_feature], axis = 1)
    pred_tmp = gnn_model(batch)
    pred_list.append(pred_tmp)

pred = torch.cat(pred_list).detach().numpy()
submission[['MLM', 'HLM']] = pred

submission.to_csv('pred/aa.csv', header = True, index = False)
# submission.to_csv('pred/vanila.csv', header = True, index = False)


# %%
