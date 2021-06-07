import numpy as np
from tqdm import tqdm
from copy import deepcopy
from collections import Counter, defaultdict
import argparse

from sklearn.utils import check_random_state
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

# ----------------------------------------------------------------------------
# System argument
# ----------------------------------------------------------------------------

#parser = argparse.ArgumentParser()
#parser.add_argument('--input', action='store', type=int, required=True)
#parser.add_argument('--id', action='store', type=int)

#args = parser.parse_args()

#print(args.input)

# ----------------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------------

class StratifiedGroupKFold():

    def __init__(self, k=10, n_repeats=1, seed=None):
        self.k = k # Number of folds
        self.n_repeats = n_repeats # Number of repeats
        self.seed = seed # Random seed for reproducibility
        
    # Implementation inspired by Kaggle
    # https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
    def split(self, X, y=None, groups=None):
        random_state = check_random_state(self.seed) # Random state generator
        
        # Repeat k-fold n_splits time with unique folds
        for repeat in range(self.n_repeats):
            labels_num = len(np.unique(y)) # Number of labels
            
            # Calculate the label distribution for each group
            y_counts_per_group = defaultdict(lambda: np.zeros(labels_num, dtype=int))
            y_distr = Counter()
            
            for label, g in zip(y, groups):
                y_counts_per_group[g][label] += 1
                y_distr[label] += 1
        
            fold_counts = defaultdict(lambda: np.zeros(labels_num, dtype=int))
            groups_per_fold = defaultdict(set)
        
            # Shuffle the groups
            label_counts = list(y_counts_per_group.items())
            random_state.shuffle(label_counts)
        
            """
            For each group and its label distribution add the group to the
            fold that would cause the least standard deviation from the
            original distribution.
            """
            
            for g, label_counts in label_counts:
                best_fold = None
                min_eval = None
                for fold in range(self.k):         
                    fold_counts[fold] += label_counts
                    std_per_label = []
                    for l in range(labels_num):
                        label_std = np.std([fold_counts[i][l] / y_distr[l] for i in range(self.k)])
                        std_per_label.append(label_std)
                        
                    fold_counts[fold] -= label_counts
                    
                    fold_eval = np.mean(std_per_label)
        
                    if min_eval == None or fold_eval < min_eval:
                        min_eval = fold_eval
                        best_fold = fold
                        
                fold_counts[best_fold] += label_counts
                groups_per_fold[best_fold].add(g)
        
            all_groups = np.unique(groups) # Get all unique groups
            for fold in range(self.k):
                train_groups = np.setdiff1d(all_groups, list(groups_per_fold[fold]))
                test_groups = list(groups_per_fold[fold])
        
                train_indices = np.where(np.isin(groups, list(train_groups)))[0]
                test_indices = np.where(np.isin(groups, list(test_groups)))[0]
                
                # Yields the indices as they are needed
                yield train_indices, test_indices
                
# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------

def format_params(params, ytrain):
    special = [key for key, value in params.items() if key.startswith('special')]
    for key in special:
        value = params[key]
        params.pop(key)
        key = key[9:]
        
        if key == 'undersamplingratio':
            majority = np.argmax([np.sum(y == l) for l in range(num_classes)])
            params['under__sampling_strategy'] = {majority : int(np.sum(ytrain == majority)*value)}
            #majority = np.argsort([np.sum(y == l) for l in range(num_classes)])
            #params['under__sampling_strategy'] = {majority[-1] : int(np.sum(ytrain == majority[-2]))}
        else:
            raise("Unknown hyperparameter")
            
    return params


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
                        Categorical(['constant', 'optimal','invscaling', 'adaptive'], name='learning_rate')]}
    }

samplers = {
    'RandomUnder' : {'sampler' : ('under', RandomUnderSampler()),
                     'space'   : []},
    'NearMiss'    : {'sampler' : ('under', NearMiss()),
                     'space'   : []},
    'RandomOver'  : {'sampler' : ('over', RandomOverSampler(sampling_strategy="not majority")),
                     'space'   : []},
    'SMOTE'       : {'sampler' : ('over', SMOTE(sampling_strategy="not majority")),
                     'space'   : []},
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
technic = 3
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
#order = np.arange(len(y))
#cut = resample(np.arange(len(y)), n_samples=100000, stratify=y, replace=False)

#X = X[cut]
#y = y[cut]
#patients = patients[cut]
#groups = groups[cut]

#%%

#import warnings
#warnings.filterwarnings("ignore", category=UserWarning)

ypred = np.empty(len(y))
kfold = StratifiedGroupKFold(k=5, n_repeats=1, seed=26062000)
pbar = tqdm(total=10*5)
    
for fold, (par_idxs, val_idxs) in enumerate(kfold.split(X, y, groups)):
    pbar.set_description("#{} fold".format(fold+1))
    Xpar, ypar = X[par_idxs], y[par_idxs]
    Xval, yval = X[val_idxs], y[val_idxs]
    
    @use_named_args(hyperspace)
    def objective(**params):
        Xtrain, Xtest, ytrain, ytest = train_test_split(Xpar, ypar,
                                                        test_size=0.2,
                                                        stratify=ypar)
        
        params = format_params(params, ytrain)
        
        pipe.set_params(**params)
        pipe.fit(Xtrain, ytrain)
        score = balanced_accuracy_score(ytest, pipe.predict(Xtest), adjusted=False)
        pbar.update(1)
        
        #np.mean(cross_val_score(reg, X, y, cv=5, n_jobs=-1,
        #                            scoring="neg_mean_absolute_error"))
        
        return -score
    
    model_opt = forest_minimize(objective, hyperspace, n_calls=10)

    params = {param.name : value for param, value in zip(hyperspace, model_opt.x)}
    params = format_params(params, ypar)
    
    pipe.set_params(**params)
    pipe.fit(Xpar, ypar)
    
    ypred[val_idxs] = pipe.predict(Xval)

#print("Best score=%.4f" % model_opt.fun)    
#print("Best parameters:", {param.name : value for param, value in zip(hyperspace, model_opt.x)})

#%%
print(classification_report(y, ypred, target_names=list(label_dict.keys())))
