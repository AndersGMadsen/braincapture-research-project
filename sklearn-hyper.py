# Classics
import numpy as np
import os
from tqdm import tqdm




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


#%%

# if window has both null and artifact, delete artifact

for i in range(len(y)):
    if np.sum(y[i]) > 1:
        y[i][5] = 0

#%%

# delete all windows with more than one artifact (not very many)

X = np.delete(X, np.where(np.sum(y, axis=1) != 1), axis=0)
G = np.delete(G, np.where(np.sum(y, axis=1) != 1), axis=0)
y = np.delete(y, np.where(np.sum(y, axis=1) != 1), axis=0)


# reverse hot-encode y

Y = np.empty(len(y), dtype=int)
for i in range(len(Y)):
    Y[i] = np.where(y[i] == 1)[0][0]
    
#%%


# Split data 
# Using groups to make sure it is on patient basis

Gidxs = resample(np.unique(G), replace=False, n_samples=int(0.8*len(np.unique(G))))

Xpar = X[np.isin(G, Gidxs)]
Ypar = Y[np.isin(G, Gidxs)]
Gpar = G[np.isin(G, Gidxs)]

Xval = X[~np.isin(G, Gidxs)]
Yval = Y[~np.isin(G, Gidxs)]
Gval = G[~np.isin(G, Gidxs)]
    

Gidxs = resample(np.unique(Gpar), replace=False, n_samples=int(0.75*len(np.unique(Gpar))))

Xtrain = Xpar[np.isin(Gpar, Gidxs)]
Ytrain = Ypar[np.isin(Gpar, Gidxs)]
Xtest = Xpar[~np.isin(Gpar, Gidxs)]
Ytest = Ypar[~np.isin(Gpar, Gidxs)]

null_idxs = np.where(Ytrain == 5)[0]

#%%

# defining model and parameter space 

model = RandomForestClassifier(n_jobs=-1, verbose=False)

space = [Integer(100, 1000, name="n_estimators"),
         Real(0.01, 0.1, name="sample_fraction"),
         Categorical([True, False], name='bootstrap'),
         Categorical(['sqrt', 'log2'], name='max_features'),
         Integer(1, 5, name="min_samples_leaf"),
         Integer(2, 10, name="min_samples_split")]

#%%
n_calls=300
pbar = tqdm(total=n_calls)

@use_named_args(space)
def objective(**params):
    sf = params.pop("sample_fraction")
    
    # Randomly down samling majority class
    null_mask = resample(null_idxs, replace=False, n_samples=int((1-sf)*len(null_idxs))) 
    
    # Fitting model
    model.fit(np.delete(Xtrain, null_mask, axis=0), np.delete(Ytrain, null_mask, axis=0))
    Ypred = model.predict(Xtest)
    balanced_accuracy = balanced_accuracy_score(Ytest, Ypred)    
    
    pbar.update(1)
    pbar.set_description("{:.3f}".format(balanced_accuracy))
    
    return -balanced_accuracy

#model_gp = gp_minimize(objective, space, n_calls=n_calls)
model_gp = forest_minimize(objective, space, n_calls=n_calls)

print()
print("Best score=%.4f" % model_gp.fun)
print("Best parameters:")
print(model_gp.x)

#%%

model.set_params(n_estimators=model_gp.x[0],
                 bootstrap=model_gp.x[2],
                 max_features=model_gp.x[3],
                 min_samples_leaf=model_gp.x[4],
                 min_samples_split=model_gp.x[5])

null_idxs = np.where(Ypar == 5)[0]
null_mask = resample(null_idxs, replace=False, n_samples=int((1-model_gp.x[1])*len(null_idxs)))

model.fit(np.delete(Xpar, null_mask, axis=0), np.delete(Ypar, null_mask, axis=0))

#%%

Ypred = model.predict(Xval)
print(classification_report(Yval, Ypred, target_names=list(label_dict.keys())))
