import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
from copy import deepcopy
import os
from scipy import stats
os.chdir(r"C:\Users\andersgm\Documents\Courses\02466 Project work F21\Project")
label_dict = {'chew': 0, 'elpp': 1, 'eyem': 2, 'musc': 3, 'shiv': 4, 'null': 5}

#%%

df = pd.read_pickle("dataframe_float32.pkl")
df["index"] = df.index

#%%
lengths = [[], [], [], [], []]

for patient_name in tqdm(np.unique(df["Name"])):
    patient = df[df["Name"] == patient_name].sort_values(by="Start", ascending=True, key=lambda col: col.values).reset_index(drop=True)
    
    start = None
    artifact = None
    for i in range(len(patient)):
        if artifact == None:    
            if patient["Multiclass label"][i] != 5:
                artifact = patient["Multiclass label"][i]
                start = patient["Start"][i]
        else:
            if patient["Multiclass label"][i] != artifact:
                lengths[artifact].append(patient["End"][i] - start)
                artifact = None
                start = None
                
                
#%%
import os

files = []
for root, dirs, files in os.walk(r"C:\Users\andersgm\Documents\Courses\02466 Project work F21\Project\Experiment\artifact_dataset\01_tcp_ar"):
    for file in files:
        if file.endswith(".edf"):
             files.append(file)


#%%
mne
mne.io.read_raw_edf(files[0])


