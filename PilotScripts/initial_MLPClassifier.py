from sklearn.metrics import f1_score




#%% Imports

# standard imports
import numpy as np
import pandas as pd
import torch
import os
from tqdm import tqdm

# plot imports
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend,
                           title, subplot, show, grid)

# Data handling
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle

from sklearn.preprocessing import StandardScaler

# Models
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

tempData_path = "/home/williamtheodor/Documents/BrainCapture/tempData/"


directories = os.listdir(tempData_path)
patients = {name.split("_")[0]: [] for name in directories}

for directory in tqdm(directories):
    name = directory.split("_")[0]
    for _, _, files in os.walk(tempData_path + directory + "/"):
        for file in files:
            patients[name].append(torch.load(tempData_path + directory + "/" + file))   
            
#%%
number_classes = 3

X = []
Y = []


for patient in patients.keys():
    temp_X, temp_y = zip(*patients[patient])

    X.append(np.array([temp_X[0].flatten().numpy() for x in temp_X]))
    temp_Y = np.empty((len(temp_y), number_classes), dtype=np.int)

    for j, label in enumerate(temp_y):
        label = np.empty(3, dtype=np.int)
        names = ['null', 'eyem', "elpp"]
        for k in range(number_classes):
            if names[k] in temp_y[j]:
                label[k] = 1
            else:
                label[k] = 0
        temp_Y[j] = label

    Y.append(temp_Y)


# %% ANN (MLP Classifier)

steps = [('scaler', StandardScaler()), ('ANN', MLPClassifier(max_iter=100000))]

pipeline = Pipeline(steps)  # define the pipeline object.

alphas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]
hs = [(20, 16), (50, 16), (100, 16), (200, 16), (20, 32), (50, 32), (100, 32), (200, 32)]

parameters = {'ANN__alpha': alphas,
              'ANN__hidden_layer_sizes': hs}

ann = GridSearchCV(pipeline, param_grid=parameters, cv=10, return_train_score=True,
                   scoring='accuracy', verbose=1)

scores = []
best_params = []
baseline_error = []
ann_error = []
test_scores = []

n_splits = len(X)

for i in range(n_splits):

    # leave one patient out CV
    X_train = X.copy()
    del X_train[i]
    X_test = X[i]
    y_train = Y.copy()
    del y_train[i]
    y_test = Y[i]

    X_train = np.array([x for patient in X_train for x in patient])
    y_train = np.array([y for patient in y_train for y in patient])

    print('split', i+1, 'of', n_splits)

    N_test = y_test.size
    ann.fit(X_train, y_train)
    ann_error.append(round(np.sum((y_test - ann.predict(X_test)) ** 2) / N_test, 2))
    baseline_error.append(round(np.sum((y_test - np.mean(y_train)) ** 2) / N_test, 2))

    best_params.append(ann.best_params_)

    test_scores.append(np.array(ann.cv_results_['mean_test_score']).reshape(len(alphas), len(hs)))

# %% Hyperparameter Plot for the ANN

plt.imshow(np.mean(test_scores, axis=0))
plt.title('RH: Hyperparameter Tuning', fontsize=18)
plt.xlabel('Size of Hidden Layers', fontsize=16)
plt.ylabel('Value of Alpha', fontsize=16)
plt.xticks(np.arange(len(hs)), [str(size) for size in hs], rotation='vertical')
plt.yticks(np.arange(len(alphas)), [str(alpha) for alpha in alphas])
plt.colorbar()
plt.show()

print('model: ANN')
print('')
print('Best paramesters: ', best_params)
print('')
print('Mean Generalization Error: ', ann_error)
print('')
print('Mean Baseline Error: ', baseline_error)


