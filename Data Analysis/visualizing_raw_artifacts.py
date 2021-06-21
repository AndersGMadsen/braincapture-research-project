#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 22:30:16 2021

@author: williamtheodor
"""
# Classics
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import json
import math


# Models
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier


# Score Metrics
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report

from sklearn.datasets import load_boston
from sklearn.ensemble import GradientBoostingRegressor


# plots
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
import matplotlib.patches as mpatches

# Fonts for pyplot
plt.rcParams['font.sans-serif'] = "Georgia"
plt.rcParams['font.family'] = "sans-serif"

# Optimization
from sklearn.model_selection import cross_val_score
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt import forest_minimize

#%%

# loading data

with open("artifacts.json", 'r') as f:
    artifacts_old = json.loads(f.read())
    
artifacts = dict()

# labeling patients 0-200 instead of whatever they were loaded as

for i, key in enumerate(artifacts_old.keys()):
    artifacts[i] = artifacts_old[key]

# stacking the sessions for each patient

for patient_id in artifacts:
    offset = 0
    
    patient = artifacts[patient_id]
    
    for session in patient:
        for row in patient[session]:

            row[0] += offset
            row[1] += offset
            
        offset += patient[session][-1][1]
        
    artifacts[patient_id] = sum(patient.values(), [])

# creating the dataframe i.e. labeling each individual second

columns = range(201)

label_dict = {'chew': 0, 'elpp': 1, 'eyem': 2, 'musc': 3, 'shiv': 4, 'null': 5}

patients = pd.DataFrame(columns=columns)

for patient_id in artifacts:
    
    patient = artifacts[patient_id]
    
    labels = []
    
    for row in patient:
        for label in range(math.ceil(row[0]), math.floor(row[1])):
            labels.append(label_dict[row[2]])
    
    while len(labels) < 10000:
        labels.append(-1)
        
    patients[patient_id] = labels
    
patients = patients.fillna(-1)

#%%

# create discrete colormap

names = ['chew', 'elpp', 'eyem', 'musc', 'shiv', 'null']
my_colors = ['white', 'yellow', 'red','lime','orange', 'purple', 'cornflowerblue']

cmap = colors.ListedColormap(my_colors)
bounds = [-1.5, -.5, .5,  1.5, 2.5, 3.5, 4.5, 5.5]
norm = colors.BoundaryNorm(bounds, cmap.N)

# legend

missing_data = mpatches.Patch(color=my_colors[0],ec='black', label='Missing data')
chew = mpatches.Patch(color=my_colors[1], label=names[0])
elpp = mpatches.Patch(color=my_colors[2],label=names[1])
eyem = mpatches.Patch(color=my_colors[3], label=names[2])
musc = mpatches.Patch(color=my_colors[4], label=names[3])
shiv = mpatches.Patch(color=my_colors[5], label=names[4])
null = mpatches.Patch(color=my_colors[6], label=names[5])
                              

# Lollipop

patientsSTEM = patients.copy()
namesSTEM = ['null', 'elpp', 'eyem', 'musc', 'shiv', 'chew']
STEM_colors = ['cornflowerblue', 'red','lime','orange', 'purple', 'yellow']


patientsSTEM = patientsSTEM.replace(5, 'null')
patientsSTEM = patientsSTEM.replace(0, 5)
patientsSTEM = patientsSTEM.replace('null', 0)

#%%

# Artifacts over time for 1 patient

# imshow plot

patient_id = 7

data = np.reshape(np.array(patients[patient_id][:70**2]), (70, 70))
    
fig, ax = plt.subplots()
ax.imshow(data, cmap=cmap, norm=norm)

plt.axis('off')


plt.legend(handles=[missing_data, chew, elpp, eyem, musc, shiv, null],
           bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.title('Artifacts over time for patient #' + str(patient_id), size=16)
plt.show()


# Lollipop plot

STEMdata = patientsSTEM[patient_id][patientsSTEM[patient_id] > -1]
x = np.linspace(0, len(STEMdata), len(STEMdata))

STEM_colorsx = [STEM_colors[int(item)] for item in STEMdata]

plt.scatter(x, STEMdata, c=STEM_colorsx)
plt.vlines(x=x, ymin=0, ymax=STEMdata, color=STEM_colorsx, alpha=0.05)
plt.yticks([0, 1, 2, 3, 4, 5], namesSTEM)

plt.legend(handles=[chew, elpp, eyem, musc, shiv, null],
           bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.title('Artifacts over time for patient #' + str(patient_id), size=16)
plt.show()



#%%

# Artifacts over time for 210 patients

#data = np.reshape(np.array(patients.loc[:100, ]), (210, 100))
data = patients.loc[0:500,].T
    
fig, ax = plt.subplots()
ax.imshow(data, cmap=cmap, norm=norm)

plt.axis('on')


plt.legend(handles=[missing_data, chew, elpp, eyem, musc, shiv, null],
           bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.title('Artifacts over time for 210 patients', size=20)
plt.ylabel('# Patient', size=16)
plt.xlabel('Seconds', size=16)

fig.savefig('myimage.png', format='png', dpi=1200)
plt.show()
