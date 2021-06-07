import numpy as np
import random
from copy import deepcopy
from collections import Counter, defaultdict
from datetime import datetime
from pickle import dump
from os.path import exists
from os import makedirs
import sys
import argparse
import warnings
import os
from tqdm import tqdm
#warnings.simplefilter("ignore", UserWarning)

from joblib import Parallel, delayed
import time

from sklearn.utils import check_random_state
#from sklearn.utils import resample
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from sklearn.multiclass import OneVsRestClassifier

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

modelname = 'LDA'
n_inner = 5
n_outer = 5
n_repeats = 1
technique = 0
seed = 123213
optimize = 10
downsample = 1

class StratifiedGroupKFold():
    def __init__(self, k=10, n_repeats=1, seed=None):
        self.k = k  # Number of folds
        self.n_repeats = n_repeats  # Number of repeats
        self.seed = seed  # Random seed for reproducibility

    # Implementation inspired by Kaggle
    # https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
    def split(self, X, y=None, groups=None):
        random_state = check_random_state(self.seed)  # Random state generator

        # Repeat k-fold n_splits time with unique folds
        for repeat in range(self.n_repeats):
            labels_num = len(np.unique(y))  # Number of labels

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

            all_groups = np.unique(groups)  # Get all unique groups
            for fold in range(self.k):
                train_groups = np.setdiff1d(all_groups, list(groups_per_fold[fold]))
                test_groups = list(groups_per_fold[fold])

                train_indices = np.where(np.isin(groups, list(train_groups)))[0]
                test_indices = np.where(np.isin(groups, list(test_groups)))[0]

                # Yields the indices as they are needed
                yield repeat, fold, train_indices, test_indices


def format_params(params, ytrain):
    special = [key for key, value in params.items() if key.startswith('special')]
    for key in special:
        value = params[key]
        params.pop(key)
        key = key[9:]

        if key == 'undersamplingratio':
            # majority = np.argmax([np.sum(y == l) for l in range(num_classes)])
            # params['under__sampling_strategy'] = {majority : int(np.sum(ytrain == majority)*value)}
            majority = np.argsort([np.sum(y == l) for l in range(num_classes)])
            params['under__sampling_strategy'] = {majority[-1]: int(np.sum(ytrain == majority[-2]))}
        else:
            raise ('Unknown hyperparameter')

    return params

hyperspace = [Categorical(categories=('svd', 'lsqr', 'eigen'), prior=None, name='model__estimator__solver')]

steps = [('under', RandomUnderSampler()),
         ('over', SMOTE(sampling_strategy='not majority', n_jobs=3)),
         ('model', OneVsRestClassifier(LinearDiscriminantAnalysis()))]

pipe = Pipeline(steps)

X = np.load(r"C:\Users\andersgm\Documents\Courses\02466 Project work F21\Project\multiclass-X.npy")
X = X.reshape(-1, np.product(X.shape[1:]))
y = np.load(r"C:\Users\andersgm\Documents\Courses\02466 Project work F21\Project\multiclass-y.npy")
num_classes = len(np.unique(y))

patients = np.load(r"C:\Users\andersgm\Documents\Courses\02466 Project work F21\Project\multiclass-patients.npy")
groups = np.empty(len(patients), dtype=int)

if downsample != -1.0:
    mask = resample(np.arange(len(y)), replace=False, n_samples=int(len(y) * downsample), stratify=y)
    X = X[mask]
    y = y[mask]
    patients = patients[mask]
    groups = groups[mask]

unique_patients = np.unique(patients)
unique_groups = np.arange(len(unique_patients))
for i, patient in enumerate(unique_patients):
    groups[patients == patient] = i

outerfold = StratifiedGroupKFold(k=n_outer, n_repeats=n_repeats, seed=seed)
innerfold = StratifiedKFold(n_splits=n_inner)

ypred = np.empty((n_repeats, len(y)), dtype=int)

for repeat, fold, par_idxs, val_idxs in tqdm(outerfold.split(X, y, groups), total=n_repeats*n_outer):
    Xpar, ypar = X[par_idxs], y[par_idxs]
    Xval = X[val_idxs]

    @use_named_args(hyperspace)
    def objective(**params):
        score = np.empty(n_inner)
        for i, (train_idxs, test_idxs) in enumerate(innerfold.split(Xpar, ypar)):
            params = format_params(params, ypar[test_idxs])
            pipe.set_params(**params)

            try:
                pipe.fit(Xpar[train_idxs], ypar[train_idxs])

            except Exception as error:
                print("Pipeline error:", error)

            score[i] = balanced_accuracy_score(ypar[test_idxs], pipe.predict(Xpar[test_idxs]), adjusted=False)

        return -np.mean(score)

    model_opt = forest_minimize(objective, hyperspace, n_calls=optimize, n_jobs=1)
    params = {param.name: value for param, value in zip(hyperspace, model_opt.x)}
    best = deepcopy(params)
    params = format_params(params, ypar)

    pipe.set_params(**params)
    pipe.fit(Xpar, ypar)
    predictions = pipe.predict(Xval)

    ypred[repeat][val_idxs] = predictions

print()