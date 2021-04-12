import numpy as np
from tqdm import tqdm

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss, ClusterCentroids, RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline 

from hyperopt import hp

from sklearn.utils import resample
from sklearn.model_selection import train_test_split

from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier


from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt import forest_minimize





#%%


#np.random.seed(26060000)
label_dict = {'chew': 0, 'elpp': 1, 'eyem': 2, 'musc': 3, 'shiv': 4, 'null': 5}

X = np.load("multiclass-X.npy")
X = X.reshape(-1, np.product(X.shape[1:]))
y = np.load("multiclass-y.npy")
patients = np.load("multiclass-patients.npy")
groups = np.empty(len(patients), dtype=int)

unique_patients = np.unique(patients)
unique_groups = np.arange(len(unique_patients))
for i, patient in enumerate(unique_patients):
    groups[patients == patient] = i

#%%
order = np.arange(len(y))
cut = resample(np.arange(len(y)), n_samples=100000, stratify=y, replace=False)

X = X[cut]
y = y[cut]
patients = patients[cut]
groups = groups[cut]


#%%
X_SMOTE, y_SMOTE = SMOTE().fit_resample(X, y)
#X_SMOTEENN, y_SMOTEENN = SMOTEENN().fit_resample(X, y)
#X_NearMiss, y_NearMiss = NearMiss().fit_resample(X, y)
#X_ClusterCentroids, y_ClusterCentroids = ClusterCentroids().fit_resample(X, y)


#%%
split = 0.2
error = 0.05
ratio_error = 0.25

for i in range(100000):
    n = np.random.randint((len(unique_groups)*(split/2)), int( len(unique_groups)*(split*2)))
    inclusive = resample(unique_groups, n_samples=n, replace=False)
    inclusive = np.where(np.isin(groups, inclusive))[0]
    
    if split*(1-error) < len(y[inclusive]) / len(y) < split*(1+error):
        for l in range(6):
            real = np.sum(y == l) / len(y)
            if not (real*(1-ratio_error) < np.sum(y[inclusive] == l) / len(inclusive) < real*(1+ratio_error)):
                break
        else:
            print(i, n)
            break
   
exclusive = np.delete(np.arange(len(y)), inclusive)
#%%

Xpar, ypar, groups_par = X[exclusive], y[exclusive], groups[exclusive]
Xval, yval, groups_val = X[inclusive], y[inclusive], groups[inclusive]

#%%

model = RandomForestClassifier(n_jobs=-1, verbose=False)

space = [Integer(100, 1000, name="n_estimators"),
         Real(0.01, 0.1, name="sample_fraction"),
         Categorical([True, False], name='bootstrap'),
         Categorical(['sqrt', 'log2'], name='max_features'),
         Integer(1, 5, name="min_samples_leaf"),
         Integer(2, 10, name="min_samples_split")]


#%%

Xtrain, Xtest, ytrain, ytest = train_test_split(Xpar, ypar, test_size=0.2, stratify=ypar)

#%%

pipe = Pipeline([("sampling", RandomUnderSampler()), ("model", RandomForestClassifier())])
#pipe.fit(X, y)

#%%

pbar = tqdm(total=10)

@use_named_args(space)
def objective(**params):
    sf = params.pop("sample_fraction")
    
    sampler = RandomUnderSampler(sampling_strategy={5: int(len(ytrain)*sf)})
    Xnew, ynew = sampler.fit_resample(Xtrain, ytrain)
    
    model.fit(Xnew, ynew)
    ypred = model.predict(Xtest)
    #score = f1_score(ytest[ytest != 5], ypred[ytest != 5], average="weighted")
    score = balanced_accuracy_score(ytest, ypred, adjusted=False   )
    
    pbar.update(1)
    
    return -score

model_gp = forest_minimize(objective, space, n_calls=10)

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


sampler = RandomUnderSampler(sampling_strategy={5: int(len(ypar)*model_gp.x[1])})
Xnew, ynew = sampler.fit_resample(Xpar, ypar)

#%%
model.fit(Xnew, ynew)

#%%

ypred = model.predict(Xval)
print(classification_report(yval, ypred, target_names=list(label_dict.keys())))


#%%
"""
X_SMOTE, y_SMOTE = SMOTE().fit_resample(X, y)
X_SMOTEENN, y_SMOTEENN = SMOTEENN().fit_resample(X, y)
X_NearMiss, y_NearMiss = NearMiss().fit_resample(X, y)
X_ClusterCentroids, y_ClusterCentroids = ClusterCentroids().fit_resample(X, y)
"""