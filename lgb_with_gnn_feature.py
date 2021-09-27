#%%
''' https://whitead.github.io/dmol-book/dl/gnn.html '''

#%%
import time
import random
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib as mpl
from tensorflow.python.client.session import _ElementFetchMapper
from tensorflow.python.ops.gen_array_ops import fingerprint

from tqdm import tqdm

import rdkit
from rdkit import Chem
from rdkit.Chem import rdDepictor, Draw

import matplotlib as mpl
import matplotlib.pyplot as plt

#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
print('TensorFlow version:', tf.__version__)
print('Eager Execution Mode:', tf.executing_eagerly())
print('available GPU:', tf.config.list_physical_devices('GPU'))
print('Num GPUs Available: ', len(tf.config.experimental.list_physical_devices('GPU')))
from tensorflow.python.client import device_lib
print('=========================================')
print(device_lib.list_local_devices())
tf.debugging.set_log_device_placement(False)

np.random.seed(100)
tf.random.set_seed(100)

#%%
warnings.filterwarnings('ignore')
sns.set_context('notebook')
sns.set_style('dark',  {'xtick.bottom':True, 'ytick.left':True, 'xtick.color': '#666666', 'ytick.color': '#666666',
                        'axes.edgecolor': '#666666', 'axes.linewidth': 0.8 , 'figure.dpi': 300})
color_cycle = ['#1BBC9B', '#F06060', '#5C4B51', '#F3B562', '#6e5687']
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=color_cycle) 
np.random.seed(0)


#%%
path = 'C:/Users/SOYOUNG/Desktop/dacon/data/'


train_tmp = pd.read_csv(path + 'train.csv', header = 0 )
train_feature_tmp = pd.read_csv(path + 'Descriptors_and_Fingerprints/train_feature.csv', header = 0)


train_drop_col = train_feature_tmp.filter(regex = 'uid|fingerprint|Fingerprint|S1_energy|T1_energy').columns
train_drop_col = list(train_drop_col) + ['VABC Volume Descriptor']
train_descriptor = train_feature_tmp.drop(train_drop_col, axis = 1)
train_obj_des = train_descriptor.dtypes[train_descriptor.dtypes == 'O'].index
train_descriptor = train_descriptor.drop(train_obj_des, axis = 1)


empty_uid = train_tmp['uid'][~train_tmp['uid'].isin(train_feature_tmp['uid'])].index
train_tmp = train_tmp.drop(empty_uid, axis = 0).reset_index(drop = True)


train_y = train_tmp['S1_energy(eV)'] - train_tmp['T1_energy(eV)']
train_smiles = train_tmp['SMILES']


train_fingerprint = np.array([])
train_fingerprint = [np.append(train_fingerprint, list(i), axis = 0) for i in 
                     tqdm(train_feature_tmp['EState fingerprint'].values)]
train_fingerprint = pd.DataFrame(train_fingerprint)
train_fingerprint = train_fingerprint.astype(int)
train_fingerprint.nunique()


train_maccs = pd.read_csv(path + 'train_fingerprint/train_maccs.csv', header = 0).iloc[:, 4:]


#%%
from mordred import Calculator, descriptors

calc = Calculator(descriptors, ignore_3D=False)
train_mor = calc.pandas([Chem.MolFromSmiles(x) for x in tqdm(train_smiles)])
train_mor.isna().sum()


#%%
test_tmp = pd.read_csv(path + 'Descriptors_and_Fingerprints/test_feature.csv', header = 0)
test_smiles = test_tmp['SMILES']


test_drop_col = test_tmp.filter(regex = 'uid|SMILES|VABC Volume Descriptor|fingerprint|Fingerprint').columns
test_descriptor = test_tmp.drop(test_drop_col, axis = 1)
test_descriptor.drop(test_descriptor[test_descriptor.columns.difference(train_descriptor.columns)].columns, axis = 1, inplace = True)


descriptor = pd.concat([train_descriptor, test_descriptor], axis = 0)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(descriptor)
# scaler.data_max_
# scaler.data_min_
train_descriptor_scaled = pd.DataFrame(scaler.transform(train_descriptor))
train_descriptor_scaled.columns = train_descriptor.columns

test_descriptor_scaled = pd.DataFrame(scaler.transform(test_descriptor))
test_descriptor_scaled.columns = test_descriptor.columns



test_fingerprint = np.array([])
test_fingerprint = [np.append(test_fingerprint, list(i), axis = 0) for i in 
                     tqdm(test_tmp['EState fingerprint'].values)]
test_fingerprint = pd.DataFrame(test_fingerprint)
test_fingerprint = test_fingerprint.astype(int)


''' fingerprint column에서 유니크 한 값이 한개인 경우 그 열 제거 '''
fingerprint = pd.concat([train_fingerprint, test_fingerprint], axis = 0)
nuniq = fingerprint.nunique()
fing_drop_idx = nuniq[nuniq == 1].index

train_fingerprint.drop(fing_drop_idx, axis = 1, inplace = True)
test_fingerprint.drop(fing_drop_idx, axis = 1, inplace = True)

train_fingerprint.columns = ['estate_' + str(i) for i in train_fingerprint.columns]
test_fingerprint.columns = ['estate_' + str(i) for i in test_fingerprint.columns]


''' 데이터 합치기 '''
train_feature = pd.concat([train_fingerprint, train_descriptor], axis = 1)
test_feature = pd.concat([test_fingerprint, test_descriptor], axis = 1)



#%%
# my_elements = {1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 
#                35: 'Br', 53: 'I'}

# def smiles2graph(sml):
#     '''Argument for the RD2NX function should be a valid SMILES sequence
#     returns: the graph
#     '''
#     m = rdkit.Chem.MolFromSmiles(sml)
#     m = rdkit.Chem.AddHs(m)
#     order_string = {rdkit.Chem.rdchem.BondType.SINGLE: 1,
#                     rdkit.Chem.rdchem.BondType.DOUBLE: 2,
#                     rdkit.Chem.rdchem.BondType.TRIPLE: 3,
#                     rdkit.Chem.rdchem.BondType.AROMATIC: 4}
#     N = len(list(m.GetAtoms()))
#     nodes = np.zeros((N, len(my_elements)))
#     lookup = list(my_elements.keys())
#     for i in m.GetAtoms():
#         nodes[i.GetIdx(), lookup.index(i.GetAtomicNum())] = 1
    
#     adj = np.zeros((N,N))
#     for j in m.GetBonds():
#         u = min(j.GetBeginAtomIdx(),j.GetEndAtomIdx())
#         v = max(j.GetBeginAtomIdx(),j.GetEndAtomIdx())        
#         order = j.GetBondType()
#         if order in order_string:
#             order = order_string[order]
#         else:
#             raise Warning('Ignoring bond order' + order)
#         adj[u, v] = 1        
#         adj[v, u] = 1
#     adj += np.eye(N)        
#     return nodes, adj


#%%
def smiles2graph(sml):
    '''Argument for the RD2NX function should be a valid SMILES sequence
    returns: the graph
    '''
    m = rdkit.Chem.MolFromSmiles(sml)
    m = rdkit.Chem.AddHs(m)
    order_string = {rdkit.Chem.rdchem.BondType.SINGLE: 1,
                    rdkit.Chem.rdchem.BondType.DOUBLE: 2,
                    rdkit.Chem.rdchem.BondType.TRIPLE: 3,
                    rdkit.Chem.rdchem.BondType.AROMATIC: 4}
    N = len(list(m.GetAtoms()))
    nodes = np.zeros((N,60))
    for i in m.GetAtoms():
        nodes[i.GetIdx(), i.GetAtomicNum()] = 1
    
    adj = np.zeros((N,N))
    for j in m.GetBonds():
        u = min(j.GetBeginAtomIdx(),j.GetEndAtomIdx())
        v = max(j.GetBeginAtomIdx(),j.GetEndAtomIdx())        
        order = j.GetBondType()
        if order in order_string:
            order = order_string[order]
        else:
            raise Warning('Ignoring bond order' + order)
        adj[u, v] = 1        
        adj[v, u] = 1
    adj += np.eye(N)
    return nodes, adj


#%%
class GCNLayer(tf.keras.layers.Layer):
    '''Implementation of GCN as layer'''
    def __init__(self, activation=None,**kwargs):
        # constructor, which just calls super constructor
        # and turns requested activation into a callable function
        super(GCNLayer, self).__init__(**kwargs)
        self.activation = K.activations.get(activation)
    
    def build(self, input_shape):
        # create trainable weights
        node_shape, adj_shape = input_shape
        self.w = self.add_weight(shape=(node_shape[2], node_shape[2]), name='w')
        
    def call(self, inputs):
        # split input into nodes, adj
        nodes, adj = inputs 
        # compute degree
        degree = tf.reduce_sum(adj, axis=-1)
        # GCN equation
        new_nodes = tf.einsum('bi,bij,bjk,kl->bil', 1 / degree, adj, nodes, self.w)
        out = self.activation(new_nodes)
        return out, adj


#%%
class GRLayer(tf.keras.layers.Layer):
    '''A GNN layer that computes average over all node features'''
    def __init__(self, name='GRLayer', **kwargs):
        super(GRLayer, self).__init__(name=name, **kwargs)
    
    def call(self, inputs):
        nodes, adj = inputs
        reduction = tf.reduce_mean(nodes, axis=1)
        return reduction
   
    
#%%
node_input = layers.Input((None, 60,))
adj_input = layers.Input((None,None,))

# GCN block
layer1 = GCNLayer('relu')
layer2 = GCNLayer('relu')
layer3 = GCNLayer('relu')

gcn_x = layer3(layer2(layer1([node_input, adj_input])))

layer5 = GRLayer()

gr_x = layer5(gcn_x)

dense1 = layers.Dense(16, 'relu')
dense2 = layers.Dense(1)

yhat = dense2(dense1(gr_x))


model = K.models.Model([node_input, adj_input], yhat)
model.summary()


#%%
def example():
    for i in range(len(train_smiles)):
        graph = smiles2graph(train_smiles[i])
        y = train_y[i]
        yield graph, y

train_data = tf.data.Dataset.from_generator(example, 
                                            output_types=((tf.float32, tf.float32), tf.float32), 
                                            output_shapes=((tf.TensorShape([None, 60]), 
                                                            tf.TensorShape([None, None])), 
                                                           tf.TensorShape([])))

# test = train_data.take(300)
# val = train_data.skip(300).take(300)
# train = train_data.skip(600)
train = train_data.take(len(train_smiles))

#%%
adam = K.optimizers.Adam(0.0001)
# nadam = K.optimizers.Nadam(0.0005)
mae = K.losses.MeanAbsoluteError()

model.compile(optimizer = adam, loss = mae, metrics = ['mse'])
# result = model.fit(train.batch(1), validation_data = val.batch(1), epochs = 15)
result = model.fit(train.batch(1), epochs = 10)

'''
    adam
    0.003 - 10, 10
    0.001 - 10, 10, 10, 10, 10
    0.0005 - 100, 40
    0.0001 - 130, 70
'''

#%%
plt.plot(result.history['loss'], label='training')
# plt.plot(result.history['val_loss'], label='validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss') 
plt.show()



#%%
# pred = model.predict(test.batch(1))

# y_test = []
# for a, b in test.take(300):
#     # g = a.numpy()
#     y_test.append(b.numpy())

# t = np.linspace(0, np.max(y_test), len(y_test))
# plt.figure(figsize=(8, 8))
# plt.scatter(y_test, pred, alpha=0.5)
# plt.plot(t, t, color='darkorange', linewidth=3)
# plt.xlabel('true data', fontsize=15)
# plt.ylabel('prediction', fontsize=15)
# plt.show()
# plt.close()

# mae(tf.squeeze(y_test), tf.squeeze(pred)).numpy()


pred = model.predict(train.batch(1))
    
t = np.linspace(0, np.max(train_y), len(train_y))
plt.figure(figsize=(8, 8))
plt.scatter(train_y, pred, alpha=0.5)
plt.plot(t, t, color='darkorange', linewidth=3)
plt.xlabel('true data', fontsize=15)
plt.ylabel('prediction', fontsize=15)
plt.show()
plt.close()

print(mae(train_y, tf.squeeze(pred)).numpy())


#%%
model.save('C:/Users/SOYOUNG/Desktop/dacon/gcn_model')

# model = K.models.load_model('C:/Users/SOYOUNG/Desktop/dacon/gcn_model')
# model.optimizer.lr
# model.layers
# model.layers[3](m.layers[2]([node[tf.newaxis, ...], adj[tf.newaxis, ...]]))



#%%
# model.layers[2]([node[tf.newaxis, ...], adj[tf.newaxis, ...]])
tf.get_logger().setLevel('ERROR')

train_gcn_feature = pd.DataFrame()
for i in tqdm(range(len(train_smiles))):
    node, adj = smiles2graph(train_smiles[i])
    # feature_tmp = pd.DataFrame(dense1(layer5(layer3(layer2(layer1([node[tf.newaxis, ...], adj[tf.newaxis, ...]]))))).numpy())
    feature_tmp = pd.DataFrame(model.layers[5](model.layers[4](model.layers[3](model.layers[2]([node[tf.newaxis, ...], adj[tf.newaxis, ...]])))).numpy())
    train_gcn_feature = pd.concat([train_gcn_feature, feature_tmp])


gcn_num = train_gcn_feature.nunique()
gcn_idx = gcn_num[gcn_num != 1].index
train_gcn_feature = train_gcn_feature[gcn_idx].reset_index(drop = True)
train_gcn_feature.columns = ['gcn_' + str(i) for i in train_gcn_feature.columns]


test_gcn_feature = pd.DataFrame()
for i in tqdm(range(len(test_smiles))):
    node, adj = smiles2graph(test_smiles[i])
    # feature_tmp = pd.DataFrame(dense1(layer5(layer3(layer2(layer1([node[tf.newaxis, ...], adj[tf.newaxis, ...]]))))).numpy())
    feature_tmp = pd.DataFrame(model.layers[5](model.layers[4](model.layers[3](model.layers[2]([node[tf.newaxis, ...], adj[tf.newaxis, ...]])))).numpy())
    test_gcn_feature = pd.concat([test_gcn_feature, feature_tmp])

test_gcn_feature = test_gcn_feature[gcn_idx].reset_index(drop = True)
test_gcn_feature.columns = ['gcn_' + str(i) for i in test_gcn_feature.columns]


gcn_feature = pd.concat([train_gcn_feature, test_gcn_feature], axis = 0)

gcn_scaler = MinMaxScaler()
gcn_scaler.fit(gcn_feature)

train_gcn_scaled = pd.DataFrame(gcn_scaler.transform(train_gcn_feature))
train_gcn_scaled.columns = train_gcn_feature.columns

test_gcn_scaled = pd.DataFrame(gcn_scaler.transform(test_gcn_feature))
test_gcn_scaled.columns = test_gcn_feature.columns




#%%
x = pd.concat([train_fingerprint, train_descriptor_scaled, train_gcn_scaled], axis = 1)


train_y = pd.DataFrame({'y': train_tmp['S1_energy(eV)'] - train_tmp['T1_energy(eV)']})
train_y['round_y'] = np.round(train_y, 1)

u, c = np.unique(train_y['round_y'], return_counts = True)
# plt.scatter(u, c)
# plt.show()
# plt.close()

y_dict = {u[i]: i for i in range(len(u))}
train_y['y_cat'] = train_y['round_y'].map(y_dict)


train_idx = random.sample(range(train_feature.shape[0]), train_feature.shape[0] - 500)
test_idx = list(set(range(train_feature.shape[0])) - set(train_idx))


# x_train = train_feature.iloc[train_idx]
x_train = x.iloc[train_idx]
y_train = train_y['y'][train_idx]

# x_test = train_feature.iloc[test_idx]
x_test = x.iloc[test_idx]
y_test = train_y['y'][test_idx]


x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)




#%%
''' NN '''
y_train = train_y['y'][train_idx]
y_test = train_y['y'][test_idx]
y_train = tf.cast(y_train, tf.float32)
y_test = tf.cast(y_test, tf.float32)


input1 = layers.Input([x.shape[1]])

dense1 = layers.Dense(100, activation = 'tanh')
dense2 = layers.Dense(50, activation = 'tanh')
dense3 = layers.Dense(20, activation = 'tanh')
dense4 = layers.Dense(1, activation = 'relu')

yhat = dense4(dense3(dense1(input1)))

nn = K.models.Model(input1, yhat)
nn.summary()


#%%
adam = K.optimizers.SGD(0.005)
mae = K.losses.MeanAbsoluteError()

nn.compile(optimizer = adam, loss = mae, metrics = ['mse'])
nn.fit(x_train, y_train, batch_size = len(y_train),epochs = 1000)


#%%
pred = nn.predict(x_test)

t = np.linspace(0, np.max(y_test), len(y_test))
plt.figure(figsize=(8, 8))
plt.scatter(y_test, pred, alpha = 0.5)
plt.plot(t, t, color='darkorange', linewidth=3)
plt.xlabel('true data', fontsize=15)
plt.ylabel('prediction', fontsize=15)
plt.show()
plt.close()

print(mae(y_test, pred).numpy())


#%% 
''' classification '''
import tensorflow_addons as tfa
from tensorflow_addons.metrics import HammingLoss

y_idx = train_y['round_y'][train_y['round_y'] <= 2].index
train_y = train_y.iloc[y_idx]

train_idx = random.sample(list(train_y.index), train_feature.shape[0] - 500)
test_idx = list(set(train_y.index) - set(train_idx))

x_train = x.iloc[train_idx]
x_test = x.iloc[test_idx]
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

y_train = train_y['y_cat'][train_idx]
y_test = train_y['y_cat'][test_idx]
y_train = tf.cast(y_train, tf.int32)
y_test = tf.cast(y_test, tf.int32)

input1 = layers.Input([x.shape[1]])

class_dense1 = layers.Dense(100, activation = 'relu')
class_dense2 = layers.Dense(50)
class_dense3 = layers.Dense(len(train_y['y_cat'].unique()), activation = 'softmax')

class_yhat = class_dense3(class_dense2(class_dense1(input1)))

classification = K.models.Model([input1], class_yhat)
classification.summary()


#%%
adam = K.optimizers.Adam(0.0001)
scc = K.losses.SparseCategoricalCrossentropy()
hamming = HammingLoss(threshold = 0.5, mode = 'multilabel')

classification.compile(optimizer = adam, loss = scc, metrics = ['accuracy', hamming])
result = classification.fit(x_train, y_train, batch_size = len(y_train), epochs = 1000)


#%%
class_pred_prob = classification.predict(x_test)
class_pred = np.argmax(class_pred_prob, -1)

t = np.linspace(0, np.max(y_test), len(y_test))
plt.figure(figsize=(8, 8))
plt.scatter(y_test, class_pred, alpha = 0.5)
plt.plot(t, t, color='darkorange', linewidth=3)
plt.xlabel('true data', fontsize=15)
plt.ylabel('prediction', fontsize=15)
plt.show()
plt.close()

print('scc: ', scc(y_test, class_pred_prob).numpy(), 
      '\nhamming loss: ', hamming(y_test, class_pred).numpy())


#%%
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

lgb = LGBMRegressor(random_state = 0, n_estimators = 1000)
# lgb.fit(x, train_y['y'])
# lgb_pred = lgb.predict(x)
# print(mean_absolute_error(train_y, lgb_pred))
lgb.fit(x_train, y_train)
lgb_pred = lgb.predict(x_test)

t = np.linspace(np.min(y_test), np.max(y_test), 100)
plt.figure(figsize=(8, 8))
plt.scatter(y_test, lgb_pred, alpha=0.5)
plt.plot(t, t, color='darkorange', linewidth=3)
plt.xlabel('true data', fontsize=15)
plt.ylabel('prediction', fontsize=15)
plt.show()
plt.close()

# t = np.linspace(np.min(train_y['y']), np.max(train_y['y']), 100)
# plt.figure(figsize=(8, 8))
# plt.scatter(train_y['y'], lgb_pred, alpha=0.01)
# plt.plot(t, t, color='darkorange', linewidth=3)
# plt.xlabel('true data', fontsize=15)
# plt.ylabel('prediction', fontsize=15)
# plt.show()
# plt.close()

print(mean_absolute_error(y_test, lgb_pred))


#%%


# x = pd.concat([test_fingerprint, test_descriptor, test_gcn_feature], axis = 1)
x = pd.concat([test_fingerprint, test_descriptor_scaled, test_gcn_scaled], axis = 1)

test_pred = lgb.predict(x)


#%%
sub = pd.read_csv(path + 'sample_submission.csv', header = 0)
sub['ST1_GAP(eV)'] = test_pred

sub.to_csv('C:/Users/SOYOUNG/Desktop/sample_submission.csv', header = True, index = False)







#%%
'''
1. 다른 그룹에서 \beta_0 + \beta_1 * x에서 \beta_0, \beta_1은 고정
2. 새로운 상수 \alpha를 더하여 \alpha만을 추정
'''

#%%
y_test = np.take(y_test, range(len(y_test)))[:, None]
indicator = np.array(list(map(lambda i: 1 if y_test[i] >= pred[i] else 0, range(len(y_test)))))
plt.figure(figsize=(8, 8))
plt.scatter(y_test[np.where(indicator == 1)], pred[np.where(indicator == 1)], alpha = 0.5)
plt.plot(t, t, color='darkorange', linewidth=3)
plt.xlabel('true data', fontsize=15)
plt.ylabel('prediction', fontsize=15)
plt.show()
plt.close()

#%%
x_A = y_test[np.where(indicator == 1)]
y_A = pred[np.where(indicator == 1)]

residual_A = y_A - x_A

sgd = K.optimizers.SGD(0.05)

alpha_A = tf.Variable(1, trainable = True, dtype = tf.float32)

for i in range(1000):
    with tf.GradientTape() as tape:
        tape.watch(alpha_A)
        loss = tf.reduce_mean(tf.square(residual_A - alpha_A))
        
        
    grad = tape.gradient(loss, [alpha_A])
    sgd.apply_gradients(zip(grad, [alpha_A]))
    
    print(i, loss)

# fitted_A = fitted.numpy()[np.where(indicator == 1)] + alpha_A


#%%
plt.figure(figsize=(8, 8))
plt.scatter(y_test[np.where(indicator == 1)], pred[np.where(indicator == 1)], alpha = 0.5)
plt.plot(t, t, color = 'darkorange', linewidth = 3)
plt.plot(t, t + alpha_A, color = 'darkorange', linewidth = 3)
plt.xlabel('true data', fontsize=15)
plt.ylabel('prediction', fontsize=15)
plt.show()
plt.close()


#%%
x_B = y_test[np.where(indicator == 0)]
y_B = pred[np.where(indicator == 0)]

residual_B = y_B - x_B

sgd = K.optimizers.SGD(0.05)

alpha_B = tf.Variable(1, trainable=True, dtype=tf.float32)

for i in range(1000):
    with tf.GradientTape() as tape:
        tape.watch(alpha_B)
        loss = tf.reduce_mean(tf.square(residual_B - alpha_B))
        
    grad = tape.gradient(loss, [alpha_B])
    sgd.apply_gradients(zip(grad, [alpha_B]))
    
    print(i, loss)

# fitted_B = fitted.numpy()[np.where(indicator == 0)] + alpha_B


#%%
plt.figure(figsize=(8, 8))
plt.scatter(y_test, pred, alpha = 0.5)
plt.plot(t, t, color = 'darkorange', linewidth = 3)
plt.plot(t, t + alpha_A, color = 'darkorange', linewidth = 3)
plt.plot(t, t + alpha_B, color = 'darkorange', linewidth = 3)
plt.xlabel('true data', fontsize=15)
plt.ylabel('prediction', fontsize=15)
plt.show()
plt.close()

#%%
from icecream import ic
ic(alpha_B, alpha_A)
# ic(alpha_B + model.weights[1][0].numpy(), alpha_A + model.weights[1][0].numpy())
# ic(beta0)


#%%
t = np.linspace(0, max(train_y), len(train_y))
indicator1 = pd.DataFrame({'ind1': [1 if train_y[i] < (t[i] + alpha_A) else 0 for i in range(len(train_y))],
                           'ind2': [1 if train_y[i] > (t[i] + alpha_B) else 0 for i in range(len(train_y))]})


