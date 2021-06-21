#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 09:22:47 2021

@author: williamtheodor
"""
# Classics
import numpy as np
import os
from tqdm import tqdm
import pandas as pd


# Models
'''
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
'''

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

# Load and filter data

label_dict = {'chew': 0, 'elpp': 1, 'eyem': 2, 'musc': 3, 'shiv': 4, 'null': 5}

X = np.load("X1.npy")
y = np.load("Y1.npy")
G = np.load("groups1.npy")

X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
nan_filter = ~np.isnan(X).any(axis=1)
X = X[nan_filter]
y = y[nan_filter]
G = G[nan_filter]


# if window has both null and artifact, delete artifact

for i in range(len(y)):
    if np.sum(y[i]) > 1:
        y[i][5] = 0


# delete all windows with more than one artifact (not very many)

X = np.delete(X, np.where(np.sum(y, axis=1) != 1), axis=0)
G = np.delete(G, np.where(np.sum(y, axis=1) != 1), axis=0)
y = np.delete(y, np.where(np.sum(y, axis=1) != 1), axis=0)


# reverse hot-encode y

Y = np.empty(len(y), dtype=int)
for i in range(len(Y)):
    Y[i] = np.where(y[i] == 1)[0][0]
    

#%%
    
patients = pd.DataFrame((Y[G==patient] for patient in range(211)), index=range(211)).T


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

patient_id = 15

data = np.reshape(np.array(patients[patient_id][:72**2]), (72, 72))
    
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

# Lollipop plots for 3 patients at a time

patient_ids = [15, 17, 18]

num_plots = len(patient_ids)
fig, axes = plt.subplots(num_plots)

fig.tight_layout(pad=2.0)

for plot in range(num_plots):
    patient_id = patient_ids[plot]
    STEMdata = patientsSTEM[patient_id][patientsSTEM[patient_id] > -1]
    x = np.linspace(0, len(STEMdata), len(STEMdata))
    
    STEM_colorsx = [STEM_colors[int(item)] for item in STEMdata]
    
    axes[plot].scatter(x, STEMdata, c=STEM_colorsx)
    
    plt.sca(axes[plot])
    plt.vlines(x=x, ymin=0, ymax=STEMdata, color=STEM_colorsx, alpha=0.05)
    plt.yticks([0, 1, 2, 3, 4, 5], namesSTEM)
    

    axes[plot].set_title('Artifacts over time for patient #' + str(patient_id), size=16)

plt.legend(handles=[chew, elpp, eyem, musc, shiv, null],
               bbox_to_anchor=(1.0, 5.0), loc='upper left')
plt.show()


#%%


# Times series of artifacts of 210 patients

patients225 = patients.copy()
for i in range(211, 225):
    patients225[i] = np.repeat(-1, 17333)


for frame in range(300, 600):
    data = np.reshape(np.array(patients225.iloc[[frame]]), (15, 15))
    
    
    fig, ax = plt.subplots()
    ax.imshow(data, cmap=cmap, norm=norm)

    plt.axis('off')
    
    
    plt.legend(handles=[missing_data, chew, elpp, eyem, musc, shiv, null],
               bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title('Time series of the Artifacts of 210 Patients', size=12)
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
plt.xlabel('# Window', size=16)
plt.show()

#%%





