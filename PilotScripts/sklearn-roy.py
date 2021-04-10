# Classics
import numpy as np
import os
from tqdm import tqdm
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Models
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


np.random.seed(26060000)

label_dict = {'chew': 0, 'elpp': 1, 'eyem': 2, 'musc': 3, 'shiv': 4, 'null': 5}

X = np.load("X1.npy")
y = np.load("Y1.npy")
groups = np.load("groups1.npy")

X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
nan_filter = ~np.isnan(X).any(axis=1)
X = X[nan_filter]
y = y[nan_filter]
groups = groups[nan_filter]


#%%
for i in range(len(y)):
    if np.sum(y[i]) > 1:
        y[i][5] = 0

#%%

X = np.delete(X, np.where(np.sum(y, axis=1) != 1), axis=0)
groups = np.delete(groups, np.where(np.sum(y, axis=1) != 1), axis=0)
y = np.delete(y, np.where(np.sum(y, axis=1) != 1), axis=0)


Y = np.empty(len(y), dtype=np.int)
for i in range(len(Y)):
    Y[i] = np.where(y[i] == 1)[0][0]


#%%
idxs = resample(np.unique(groups), replace=False, n_samples=50, random_state=1928)

Xtrain = X[~np.isin(groups, idxs)]
Ytrain = Y[~np.isin(groups, idxs)]

Xtest = X[np.isin(groups, idxs)]
Ytest = Y[np.isin(groups, idxs)]

#%%
mask = resample(np.where(Ytrain == 5)[0], replace=False, n_samples=509831 - 20000)

Xtraindown = np.delete(Xtrain, mask, axis=0)
Ytraindown = np.delete(Ytrain, mask, axis=0)

#%%

scaler = StandardScaler()
scaler.fit(Xtraindown)
Xtraindown = scaler.transform(Xtraindown)
Xtest = scaler.transform(Xtest)

#%%

models = [RandomForestClassifier(), MLPClassifier(), LinearDiscriminantAnalysis(), GaussianNB(), KNeighborsClassifier(), SGDClassifier()]

#%%
for model in models:
    model.fit(Xtraindown, Ytraindown)
    Ypred = model.predict(Xtest)
 
    print(classification_report(Ytest, Ypred, target_names=list(label_dict.keys())))