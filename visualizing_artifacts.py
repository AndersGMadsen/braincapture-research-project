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


# Optimization
from sklearn.model_selection import cross_val_score
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt import forest_minimize



def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

np.random.seed(26060000)

label_dict = {'chew': 0, 'elpp': 1, 'eyem': 2, 'musc': 3, 'shiv': 4, 'null': 5}


# Load and filter data

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

#%%


patients[[0, 1, 2, 3, 4, 5]].plot()


#%% 


patient = patients[1]
sqrt = 
while len(patient)

data = np.reshape(np.array(patient.dropna()[:53**2]), (53, 53))

# create discrete colormap
cmap = colors.ListedColormap(['yellow', 'red','lime',  'orange', 'purple', 'cornflowerblue'])
bounds = [0,1, 2, 3, 4, 5]
norm = colors.BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots()
ax.imshow(data, cmap=cmap, norm=norm)

# draw gridlines
ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=2)
plt.axis('off')

plt.show()
