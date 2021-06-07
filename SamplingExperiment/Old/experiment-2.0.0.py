
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score

# Sklearn Optimize
from skopt.space import Real, Integer, Categorical
from skopt import gp_minimize, forest_minimize
from skopt.utils import use_named_args

# Imbalance Learn
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import NearMiss, RandomUnderSampler, EditedNearestNeighbours
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.combine import SMOTEENN


# Models
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

models = {    
    'ADA' : {'model' : ('model', AdaBoostClassifier()),
             'space' : [Integer(50, 200, name='n_estimators'),
                        Real(0.5, 2, name='learning_rate'),
                        Categorical(['SAMME', 'SAMME.R'], name='algorithm')]},
             
    'GNB' : {'model' : ('model', GaussianNB()),
             'space' : [Real(10**(-9), 10**0, prior='log-uniform', name='var_smoothing')]},
    
    'KNN' : {'model' : ('model', KNeighborsClassifier(n_jobs=1)),
             'space' : [Integer(5, 20, name='n_neighbors'),
                        Categorical(['auto', 'ball_tree', 'kd_tree', 'brute'], name='algorithm'),
                        Integer(15, 60, name='leaf_size'),
                        Integer(1, 2, name='p')]},
    
    'LDA' : {'model' : ('model', LinearDiscriminantAnalysis()),
             'space' : [Categorical(['svd', 'lsqr', 'eigen'], name='solver')]},
    
    'MLP' : {'model' : ('model', MLPClassifier()),
             'space' : [Categorical(['identity', 'logistic', 'tanh', 'relu'], name='activation'),
                        Real(10**(-9), 10**0, prior='log-uniform', name='alpha'),
                        Categorical(['constant', 'invscaling', 'adaptive'], name='learning_rate')]},
    
    
    'RFO' : {'model' : ('model', RandomForestClassifier(n_jobs=-1)),
             'space' : [Integer(100, 1000, name='n_estimators'),
                        Categorical(['sqrt', 'log2'], name='max_features'),
                        Integer(1, 5, name='min_samples_leaf'),
                        Integer(2, 10, name='min_samples_split')]},
    
    'SGD' : {'model' : ('model', SGDClassifier(n_jobs=1)),
             'space' : [Real(10**(-9), 10**0, prior='log-uniform', name='alpha'),
                        Categorical(['l2', 'l1', 'elasticnet'], name='penalty'),
                        Real(10**(-4), 10**(-2), prior='log-uniform', name='tol'),
                        Categorical(['constant', 'optimal', 'invscaling', 'adaptive'], name='learning_rate')]}
    }

samplers = {
    'RandomUnder'   : {'sampler'    : ('under', RandomUnderSampler()),
                       'space'      : []},
    'NearMiss'      : {'sampler'    : ('under', NearMiss()),
                       'space'      : []},
    'RandomOver'    : {'sampler'    : ('over', RandomOverSampler(sampling_strategy="not majority")),
                       'space'      : []},
    'SMOTE'         : {'sampler'    : ('over', SMOTE(sampling_strategy="not majority")),
                       'space'      : []},
    }

technics = [
        [samplers['RandomUnder']],
        [samplers['NearMiss']],
        [samplers['RandomUnder'], samplers["RandomOver"]],
        [samplers["NearMiss"], samplers['RandomOver']],
        [samplers['RandomUnder'], samplers['SMOTE']],
        [samplers['NearMiss'], samplers['SMOTE']]
    ]

modelname = 'LDA'
technic = 4
hyperspace = []
steps = []

for parameter in deepcopy(models[modelname]['space']):
    parameter.name = 'model__' + parameter.name
    hyperspace.append(parameter)

for sampler in deepcopy(technics[technic]):
    steps.append(sampler['sampler'])
    for parameter in sampler['space']:
        parameter.name = sampler['sampler'][0] + '__' + parameter.name
        hyperspace.append(parameter)
        
steps.append(models[modelname]['model'])

if 'under' in list(zip(*steps))[0]:
    hyperspace.append(Real(0.0, 0.05, name='special__undersamplingratio'))

pipe = Pipeline(steps)

#%%

#np.random.seed(26060000)
label_dict = {'chew': 0, 'elpp': 1, 'eyem': 2, 'musc': 3, 'shiv': 4, 'null': 5}
num_classes = 6

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

def update_params(params, ytrain):
    special = [key for key, value in params.items() if key.startswith('special')]
    for key in special:
        value = params[key]
        params.pop(key)
        key = key[9:]
        
        if key == 'undersamplingratio':
            majority = np.argmax([np.sum(y == l) for l in range(num_classes)])
            params['under__sampling_strategy'] = {majority : int(np.sum(ytrain == l)*value)}
            print(params)
        else:
            print("Unknown hyperparameter")
    return params

#import warnings
#warnings.filterwarnings("ignore", category=UserWarning)

pbar = tqdm(total=50)

@use_named_args(hyperspace)
def objective(**params):
    Xtrain, Xtest, ytrain, ytest = train_test_split(Xpar, ypar, test_size=0.2, stratify=ypar)
    
    params = update_params(params, ytrain)
    
    pipe.set_params(**params)
    pipe.fit(Xtrain, ytrain)
    ypred = pipe.predict(Xtest)
    score = balanced_accuracy_score(ytest, ypred, adjusted=False)
    pbar.update(1)
    
    return -score

model_opt = forest_minimize(objective, hyperspace, n_calls=50)

print()
print("Best score=%.4f" % model_opt.fun)    
print("Best parameters:", {param.name : value for param, value in zip(hyperspace, model_opt.x)})

#%%
params = {param.name : value for param, value in zip(hyperspace, model_opt.x)}
params = update_params(params, ypar)

pipe.set_params(**params)
pipe.fit(Xpar, ypar)

ypred = pipe.predict(Xval)
print(classification_report(yval, ypred, target_names=list(label_dict.keys())))
