import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
from copy import deepcopy
import os
from scipy import stats
from sklearn.metrics import classification_report
os.chdir(r"C:\Users\andersgm\Documents\Courses\02466 Project work F21\Project")
label_dict = {'chew': 0, 'elpp': 1, 'eyem': 2, 'musc': 3, 'shiv': 4, 'null': 5}

#%%

df = pd.read_pickle("dataframe_float32.pkl")
df["index"] = df.index

ypreds = np.load("predictions_LDA_4_5_5_5_55784899_04-06-21_20-08-15.npy")

#%%
def clean(a):
    a = deepcopy(a)
    
    for n in range(1, 2):
        for i in range(n+1, len(a)-n*2-1):
            if np.all(a[i-n-1:i] == a[i-n-1]) and np.all(a[i-n-1] == a[i+n:i+2*n+1]):
                a[i:i+n] = a[i-n-1]
            
    for n in range(4, 7):
        mask = np.zeros(len(a))
        for i in range(len(a)-n):
            if np.all(a[i:i+n] == a[i]):
                mask[i:i+n] = True
                
        for i in range(len(a)):
            if not mask[i]:
                a[i] = 5    
    return a

def clean2(a):
    a = deepcopy(a)
    for i in range(2, len(a)-2):
        if a[i-2] == a[i-1] and a[i-1] == a[i+1] and a[i+1] == a[i+2]:
            a[i] = a[i-1]
        
    for i in range(3, len(a)-4):
        if a[i-3] == a[i-2] and a[i-2] == a[i-1] and a[i-1] == a[i+2] and a[i+2] == a[i+3] and a[i+3] == a[i+4]:
            a[i] = a[i-1]
            a[i+1] = a[i-1]
            
    mask = np.zeros(len(a))
    for i in range(len(a)-4):
        if np.all(a[i:i+4] == a[i]):
            mask[i:i+4] = 1
            
    for i in range(len(a)):
        if not mask[i]:
            a[i] = 5
    
    mask = np.zeros(len(a))
    for i in range(len(a)-5):
        if np.all(a[i:i+5] == a[i]):
            mask[i:i+5] = 1
            
    for i in range(len(a)):
        if not mask[i]:
            a[i] = 5
            
    mask = np.zeros(len(a)) 
    for i in range(len(a)-6):
        if np.all(a[i:i+6] == a[i]):
            mask[i:i+6] = 1
            
    for i in range(len(a)):
        if not mask[i]:
            a[i] = 5
    
    mask = np.zeros(len(a)) 
    for i in range(len(a)-7):
        if np.all(a[i:i+7] == a[i]):
            mask[i:i+7] = 1
            
    for i in range(len(a)):
        if not mask[i]:
            a[i] = 5
            
            
    for i in range(4, len(a)-6):
        if np.all(a[i-7:i] == a[i-7]) and np.all(a[i-7] == a[i+6:i+6+7]):
            a[i:i+6] = a[i-7]
            
    return a


#%%
    
filtered_ypreds = np.empty(ypreds.shape)
filtered_ypreds.fill(-1)

for patient_name in tqdm(np.unique(df["Name"])):
    patient = df[df["Name"] == patient_name].sort_values(by="Start", ascending=True, key=lambda col: col.values).reset_index(drop=True)
    
    for i in range(5):
        filtered_ypreds[i][patient["index"].values] = clean(ypreds[i][patient["index"].values])

filtered_ypreds2 = np.empty(ypreds.shape)
filtered_ypreds2.fill(-1)

for patient_name in tqdm(np.unique(df["Name"])):
    patient = df[df["Name"] == patient_name].sort_values(by="Start", ascending=True, key=lambda col: col.values).reset_index(drop=True)
    
    for i in range(5):
        filtered_ypreds2[i][patient["index"].values] = clean2(ypreds[i][patient["index"].values])
    

#%%
print(classification_report(np.tile(df["Multiclass label"].values, ypreds.shape[0]), ypreds.flatten(), target_names=list(label_dict.keys())))
print(["{:.3f}".format(np.mean(ypreds.flatten()[np.tile(df["Multiclass label"].values, ypreds.shape[0]) == i] == i)) for i in range(6)])

#%%
print(classification_report(np.tile(df["Multiclass label"].values, filtered_ypreds.shape[0]), filtered_ypreds.flatten(), target_names=list(label_dict.keys())))
print(["{:.3f}".format(np.mean(filtered_ypreds.flatten()[np.tile(df["Multiclass label"].values, filtered_ypreds.shape[0]) == i] == i)) for i in range(6)])

#%%
print(classification_report(np.tile(df["Multiclass label"].values, filtered_ypreds2.shape[0]), filtered_ypreds.flatten(), target_names=list(label_dict.keys())))
print(["{:.3f}".format(np.mean(filtered_ypreds.flatten()[np.tile(df["Multiclass label"].values, filtered_ypreds.shape[0]) == i] == i)) for i in range(6)])
