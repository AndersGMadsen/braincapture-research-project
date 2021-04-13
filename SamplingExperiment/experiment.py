import numpy as np
import random
from tqdm import tqdm
from copy import deepcopy
from collections import Counter, defaultdict
from datetime import datetime
from pickle import dump
from os.path import exists
from os import makedirs
import argparse
import warnings

from sklearn.utils import check_random_state
from sklearn.utils import resample
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

# Sklearn Optimize
from skopt.space import Real, Integer, Categorical
from skopt import forest_minimize #gp_minimize
from skopt.utils import use_named_args

# Imbalance Learn
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import NearMiss, RandomUnderSampler #EditedNearestNeighbours
from imblearn.over_sampling import SMOTE, RandomOverSampler
#from imblearn.combine import SMOTEENN

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
if False:
    parser = argparse.ArgumentParser()
    parser.add_argument('--x', action='store', type=str, required=True)
    parser.add_argument('--y', action='store', type=str, required=True)
    parser.add_argument('--groups', action='store', type=str, required=True)
    parser.add_argument('--model', action='store', type=str, required=True)
    parser.add_argument('--technique', action='store', type=int, required=True)
    parser.add_argument('--verbose', action='store', type=bool, required=False, default=True)
    parser.add_argument('--seed', action='store', type=int, required=False, default=None)
    parser.add_argument('--n_outer', action='store', type=int, required=False, default=5)
    parser.add_argument('--n_inner', action='store', type=int, required=False, default=5)
    parser.add_argument('--n_repeats', action='store', type=int, required=False, default=1)
    parser.add_argument('--n_jobs', action='store', type=int, required=False, default=1)
    parser.add_argument('--outdir', action='store', type=str, required=False, default='results')
    parser.add_argument('--optimize', action='store', type=int, required=False, default=50)
    parser.add_argument('--use', action='store', type=float, required=False, default=1)
    
    args = parser.parse_args()
    
    # print(args.input)
    x_path = args.x
    y_path = args.y
    groups_path = args.groups
    verbose = args.verbose
    seed = args.seed
    n_inner = args.n_inner
    n_outer = args.n_outer
    n_repeats = args.n_repeats
    n_jobs = args.n_jobs
    modelname = args.model
    technique = args.technique
    outdir = args.outdir
    use = args.use
    optimize = args.optimize

else:
    x_path = "multiclass-X.npy"
    y_path = "multiclass-y.npy"
    groups_path = "multiclass-patients.npy"
    verbose = True
    seed = 260600
    n_inner = 2
    n_outer = 2
    n_repeats = 1
    n_jobs = -1
    modelname = "LDA"
    technique = 0
    outdir = "results"
    use = 0.1
    optimize=10
    
np.random.seed(seed)
random.seed(seed)
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
        
            '''
            For each group and its label distribution add the group to the
            fold that would cause the least standard deviation from the
            original distribution.
            '''
            
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
            raise('Unknown hyperparameter')
            
    return params


models = {    
    'ADA' : {'model' : ('model', AdaBoostClassifier()),
             'space' : [Integer(50, 200, name='n_estimators'),
                        Real(0.5, 2, name='learning_rate'),
                        Categorical(['SAMME', 'SAMME.R'], name='algorithm')]},
             
    'GNB' : {'model' : ('model', GaussianNB()),
             'space' : [Real(10**(-9), 10**0, prior='log-uniform', name='var_smoothing')]},
    
    'KNN' : {'model' : ('model', KNeighborsClassifier(n_jobs=n_jobs)),
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
    
    
    'RFO' : {'model' : ('model', RandomForestClassifier(n_jobs=n_jobs)),
             'space' : [Integer(100, 1000, name='n_estimators'),
                        Categorical(['sqrt', 'log2'], name='max_features'),
                        Integer(1, 5, name='min_samples_leaf'),
                        Integer(2, 10, name='min_samples_split')]},
    
    'SGD' : {'model' : ('model', SGDClassifier(n_jobs=n_jobs)),
             'space' : [Real(10**(-9), 10**0, prior='log-uniform', name='alpha'),
                        Categorical(['l2', 'l1', 'elasticnet'], name='penalty'),
                        Real(10**(-4), 10**(-2), prior='log-uniform', name='tol'),
                        Categorical(['constant', 'optimal','invscaling', 'adaptive'], name='learning_rate')]}
    }

samplers = {
    'RandomUnder' : {'sampler' : ('under', RandomUnderSampler()),
                     'space'   : []},
    'NearMiss'    : {'sampler' : ('under', NearMiss(n_jobs=n_jobs)),
                     'space'   : []},
    'RandomOver'  : {'sampler' : ('over', RandomOverSampler(sampling_strategy='not majority')),
                     'space'   : []},
    'SMOTE'       : {'sampler' : ('over', SMOTE(sampling_strategy='not majority', n_jobs=n_jobs)),
                     'space'   : []},
    }

techniques = [
        [samplers['RandomUnder']],
        [samplers['NearMiss']],
        [samplers['RandomUnder'], samplers['RandomOver']],
        [samplers['NearMiss'], samplers['RandomOver']],
        [samplers['RandomUnder'], samplers['SMOTE']],
        [samplers['NearMiss'], samplers['SMOTE']]
    ]

hyperspace = []
steps = []

for parameter in deepcopy(models[modelname]['space']):
    parameter.name = 'model__' + parameter.name
    hyperspace.append(parameter)

for sampler in deepcopy(techniques[technique]):
    steps.append(sampler['sampler'])
    for parameter in sampler['space']:
        parameter.name = sampler['sampler'][0] + '__' + parameter.name
        hyperspace.append(parameter)
        
steps.append(models[modelname]['model'])

if 'under' in list(zip(*steps))[0]:
    hyperspace.append(Real(0.0, 0.05, name='special__undersamplingratio'))

pipe = Pipeline(steps)


#%%

X = np.load(x_path)
X = X.reshape(-1, np.product(X.shape[1:]))
y = np.load(y_path)
num_classes = len(np.unique(y))

patients = np.load(groups_path)
groups = np.empty(len(patients), dtype=int)

unique_patients = np.unique(patients)
unique_groups = np.arange(len(unique_patients))
for i, patient in enumerate(unique_patients):
    groups[patients == patient] = i

#%%

order = np.arange(len(y))
cut = resample(np.arange(len(y)), n_samples=int(len(y)*use), stratify=y, replace=False)

X = X[cut]
y = y[cut]
patients = patients[cut]
groups = groups[cut]

#%%
ypred = np.empty(len(y), dtype=int)
outerfold = StratifiedGroupKFold(k=n_outer, n_repeats=n_repeats, seed=seed)
innerfold = StratifiedKFold(n_splits=n_inner)
if verbose: pbar = tqdm(total=n_repeats*n_inner*n_outer*optimize)

best_hyperparametes = {}

ypred = np.empty((n_repeats, len(y)))

for fold, (par_idxs, val_idxs) in enumerate(outerfold.split(X, y, groups)):
    if verbose: pbar.set_description('#{} fold'.format(fold % n_outer))
    
    Xpar, ypar = X[par_idxs], y[par_idxs]
    Xval, yval = X[val_idxs], y[val_idxs]
    
    @use_named_args(hyperspace)
    def objective(**params):
        score = np.empty(n_inner)
        for i, (train_idxs, test_idxs) in enumerate(innerfold.split(Xpar, ypar)):
            params = format_params(params, ypar[test_idxs])
            pipe.set_params(**params)
            
            for _ in range(10):
                try:
                    pipe.fit(Xpar[train_idxs], ypar[train_idxs])
                    break
                except:
                    pass
            else:
                try:
                    pipe.fit(Xpar[train_idxs], ypar[train_idxs])
                except Exception as error:
                    warnings.warn(str(error), RuntimeWarning)
                    
                
            score[i] = balanced_accuracy_score(ypar[test_idxs], pipe.predict(Xpar[test_idxs]), adjusted=False)
        
            if verbose: pbar.update(1)
  
        return -np.mean(score)
    
    model_opt = forest_minimize(objective, hyperspace, n_calls=optimize, n_jobs=n_jobs)

    params = {param.name : value for param, value in zip(hyperspace, model_opt.x)}
    best_hyperparametes[fold] = deepcopy(params)
    params = format_params(params, ypar)
    
    pipe.set_params(**params)
    pipe.fit(Xpar, ypar)
    
    ypred[fold // n_outer][val_idxs] = pipe.predict(Xval)

#%%

if not exists(outdir):
    makedirs(outdir)
    
now = datetime.now().strftime('%d-%m-%y_%H-%M-%S')
prediction_name = 'predictions_{}_{}_{}_{}_{}_{}_{}'.format(modelname, technique, n_repeats, n_outer, n_inner, seed, now) 
np.save(outdir + '/' + prediction_name, ypred, allow_pickle=True)
hyperparameter_name = 'hyperparameters_{}_{}_{}_{}_{}_{}_{}'.format(modelname, technique, n_repeats, n_outer, n_inner, seed, now)
with open(outdir + '/' + hyperparameter_name + '.pkl', 'wb') as file:
    dump(best_hyperparametes, file)
#%%
label_dict = {'chew': 0, 'elpp': 1, 'eyem': 2, 'musc': 3, 'shiv': 4, 'null': 5}
print(classification_report(np.tile(y, n_repeats), ypred.flatten(), target_names=list(label_dict.keys())))
