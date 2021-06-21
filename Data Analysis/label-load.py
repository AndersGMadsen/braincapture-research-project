import numpy as np
import torch
from tqdm import tqdm
import os

tempData_path = "C:/Users/AndersGM/Documents/Courses/02466 Project work F21/Project/Experiment/artifact_dataset/01_tcp_ar/"


directories = os.listdir(tempData_path)


#%%
label_files = []

for directory in directories:
    for root, _, files in os.walk(tempData_path + directory + "/"):
        for file in files:
            if file.split(".")[1] == 'tse' and file.count('_') == 2:
                label_files.append((root + "/" + file).replace("\\", "/"))
                
#%%

artifacts = {}

for label_file in label_files:
    with open(label_file, 'r') as file:
        temp = label_file.split("/")[-1].split('_')
        patient = temp[0]
        session = temp[1]        
        
        if not patient in artifacts:
            artifacts[patient] = {}
        
        artifacts[patient][session] = []
        text = file.read().split('\n')[2:-1]
        for t in text:
            start, end, artifact = t.split(' ')[:3]
            start = float(start)
            end = float(end)
            artifacts[patient][session].append((start, end, artifact))
        
        
        
#%%

import json

dump = json.dumps(artifacts)
with open("artifacts.json", 'w') as f:
    f.write(dump)
