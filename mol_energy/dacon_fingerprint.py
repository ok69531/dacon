#%%
import pandas as pd
import numpy as np

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, MACCSkeys, RDKFingerprint


#%%
train = pd.read_csv('C:/Users/SOYOUNG/Desktop/dacon/data/train.csv', header = 0)
test = pd.read_csv('C:/Users/SOYOUNG/Desktop/dacon/data/test.csv', header = 0)

train_mol = [Chem.MolFromSmiles(i) for i in tqdm(train['SMILES'])]
train_na_idx = [i for i in range(len(train_mol)) if train_mol[i] == None]
len(train_na_idx)


test_mol = [Chem.MolFromSmiles(i) for i in tqdm(test['SMILES'])]
test_na_idx = [i for i in range(len(test_mol)) if test_mol[i] == None]
len(test_na_idx)


#%%
# https://www.rdkit.org/docs/GettingStartedInPython.html
''' train '''
# ---------------- MACCS Keys ---------------- #

train_macc = [MACCSkeys.GenMACCSKeys(i) for i in tqdm(train_mol)]
train_macc_bit = [i.ToBitString() for i in tqdm(train_macc)]

train_maccs_fp = pd.DataFrame(list(train_macc_bit[0])).transpose()

for i in tqdm(range(1, len(train_macc_bit))):
    maccs_tmp = pd.DataFrame(list(train_macc_bit[i])).transpose()
    train_maccs_fp = pd.concat([train_maccs_fp, maccs_tmp], ignore_index = True)


train_maccs = pd.concat([train, train_maccs_fp], axis = 1)
train_maccs.isna().sum().sum()

train_maccs.to_csv('C:/Users/SOYOUNG/Desktop/dacon/data/train_fingerprint/train_maccs.csv', index = False, header = True)


#%%

# ---------------- Topological Fingerprint (TF) ---------------- #

train_tf = [Chem.RDKFingerprint(i) for i in tqdm(train_mol)]
train_tf_bit = [i.ToBitString() for i in tqdm(train_tf)]

train_tfs_tmp = pd.DataFrame(list(train_tf_bit[0])).transpose()

for i in tqdm(range(train_tfs_tmp.shape[0], len(train_tf_bit))):
    tfs_tmp = pd.DataFrame(list(train_tf_bit[i])).transpose()
    train_tfs_tmp = pd.concat([train_tfs_tmp, tfs_tmp], ignore_index = True)


train_tfs = pd.concat([train, train_tfs_tmp], axis = 1)
train_tfs.isna().sum().sum()

train_tfs.to_csv('C:/Users/SOYOUNG/Desktop/dacon/data/train_fingerprint/train_tfs.csv', header = True, index = False)


#%%

# ---------------- Morgan ---------------- # > smilar to ECFP/FCFP

train_mor = [AllChem.GetMorganFingerprintAsBitVect(i, 2, nBits=1024) for i in tqdm(train_mol)]
train_mor_bit = [i.ToBitString() for i in tqdm(train_mor)]

train_morgan_tmp = pd.DataFrame(list(train_mor_bit[0])).transpose()

for i in tqdm(range(1, len(train_mor))):
    morgan_tmp = pd.DataFrame(list(train_mor_bit[i])).transpose()
    train_morgan_tmp = pd.concat([train_morgan_tmp, morgan_tmp], ignore_index = True)

train_morgan = pd.concat([train, train_morgan_tmp], axis = 1)
train_morgan.isna().sum().sum()

train_morgan.to_csv('C:/Users/SOYOUNG/Desktop/dacon/data/train_fingerprint/train_morgan.csv', header = True, index = False)

#%%

# ----------------  ---------------- #


#%%
''' test '''
# ---------------- MACCS Keys ---------------- #

test_macc = [MACCSkeys.GenMACCSKeys(i) for i in tqdm(test_mol)]
test_macc_bit = [i.ToBitString() for i in tqdm(test_macc)]

test_maccs_fp = pd.DataFrame(list(test_macc_bit[0])).transpose()

for i in tqdm(range(1, len(test_macc_bit))):
    maccs_tmp = pd.DataFrame(list(test_macc_bit[i])).transpose()
    test_maccs_fp = pd.concat([test_maccs_fp, maccs_tmp], ignore_index = True)


test_maccs = pd.concat([test, test_maccs_fp], axis = 1)
test_maccs.isna().sum().sum()

test_maccs.to_csv('C:/Users/SOYOUNG/Desktop/dacon/data/test_fingerprint/test_maccs.csv', index = False, header = True)


#%%

# ---------------- Topological Fingerprint (TF) ---------------- #

test_tf = [Chem.RDKFingerprint(i) for i in tqdm(test_mol)]
test_tf_bit = [i.ToBitString() for i in tqdm(test_tf)]

test_tfs_tmp = pd.DataFrame(list(test_tf_bit[0])).transpose()

for i in tqdm(range(1, len(test_tf_bit))):
    tfs_tmp = pd.DataFrame(list(test_tf_bit[i])).transpose()
    test_tfs_tmp = pd.concat([test_tfs_tmp, tfs_tmp], ignore_index = True)

test_tfs = pd.concat([test, test_tfs_tmp], axis = 1)
test_tfs.isna().sum().sum()

test_tfs.to_csv('C:/Users/SOYOUNG/Desktop/dacon/data/test_fingerprint/test_tfs.csv', header = True, index = False)


#%%

# ---------------- Morgan ---------------- # > smilar to ECFP/FCFP

test_mor = [AllChem.GetMorganFingerprintAsBitVect(i, 2, nBits=1024) for i in tqdm(test_mol)]
test_mor_bit = [i.ToBitString() for i in tqdm(test_mor)]

test_morgan_tmp = pd.DataFrame(list(test_mor_bit[0])).transpose()

for i in tqdm(range(1, len(test_mor))):
    morgan_tmp = pd.DataFrame(list(test_mor_bit[i])).transpose()
    test_morgan_tmp = pd.concat([test_morgan_tmp, morgan_tmp], ignore_index = True)

test_morgan = pd.concat([test, test_morgan_tmp], axis = 1)
test_morgan.isna().sum().sum()

test_morgan.to_csv('C:/Users/SOYOUNG/Desktop/dacon/data/test_fingerprint/test_morgan.csv', header = True, index = False)
