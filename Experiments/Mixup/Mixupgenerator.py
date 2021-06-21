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


from sklearn.utils import check_random_state
# from sklearn.utils import resample
from sklearn.metrics import classification_report, balanced_accuracy_score, fbeta_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample

# Sklearn Optimize
from skopt.space import Real, Integer, Categorical
from skopt import forest_minimize  # gp_minimize
from skopt.utils import use_named_args

# Imbalance Learn
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler

#import mixup modules

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.utils.extmath import softmax
from sklearn.metrics import classification_report, balanced_accuracy_score, fbeta_score
#import matplotlib.pyplot.plot as plt
import numpy as np
from matplotlib import pyplot
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.patches as mpatches



from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
# Score Metrics
from sklearn.metrics import balanced_accuracy_score
from imblearn.under_sampling import RandomUnderSampler

import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


# Mixing two training data points and returning the mixed data point and its label between 0 and 1
def mix(eeg_info_1=None, eeg_info_2=None, lam=None):
    X1, X2 = eeg_info_1["eeg"], eeg_info_2["eeg"]

    y1, y2 = eeg_info_1["label"], eeg_info_2["label"]

    mix_X = lam * X1 + (1. - lam) * X2
    label = lam * y1 + (1. - lam) * y2

    return mix_X, label


# Takes a training set (and labels) as input and returns the new mixup data points
def mixup(X_train=None, y_train=None, g_train=None, size=None):
    ohe = OneHotEncoder()
    ohe.fit([[0], [1], [2], [3], [4], [5]])
    y_train_soft = ohe.transform(y_train.reshape(-1, 1)).toarray()

    classes = [0, 1, 2, 3, 4, 5]

    N = size  # Number of mixup augmentation examples
    c = 10  # sampling times
    n = int(N / c)  # points for each sampling

    shuffler = np.random.permutation(len(y_train))
    X_train, y_train, y_train_soft, g_train = X_train[shuffler], y_train[shuffler], y_train_soft[shuffler], g_train[
        shuffler]

    # X_train_original, y_train_original, y_train_soft_original = X_train, y_train, y_train_soft
    # X_train_original, y_train_original = X_train, y_train

    mix_Xs = np.empty(shape=(0, X_train.shape[1]), dtype=float)
    mix_ys = np.array([], dtype=int)
    mix_ys_soft = np.empty(shape=(0, y_train_soft.shape[1]), dtype=float)
    # extra data for analysis
    y_1s, y_2s = np.array([], dtype=int), np.array([], dtype=int)
    lams, g1s, g2s = np.array([], dtype=float), np.array([], dtype=int), np.array([], dtype=int)

    for i in tqdm(range(c)):

        class1 = int(np.random.choice(classes))

        ind1 = np.where(y_train == class1)[0]
        ind2 = np.where(y_train != class1)[0]
        ind1, ind2 = shuffle(ind1), shuffle(ind2)
        idx1, idx2 = ind1[:n], ind2[:n]

        # create minibatch
        X_1, X_2 = X_train[idx1], X_train[idx2]
        y_1, y_2 = y_train[idx1], y_train[idx2]
        y_1_soft, y_2_soft = y_train_soft[idx1], y_train_soft[idx2]
        g_1, g_2 = g_train[idx1], g_train[idx2]

        y_1s, y_2s = np.append(y_1s, y_1), np.append(y_2s, y_2)
        g1s, g2s = np.append(g1s, g_1), np.append(g2s, g_2)

        for j in tqdm(range(n)):
            eeg_info_1, eeg_info_2 = {"eeg": X_1[j], "label": y_1_soft[j]}, {"eeg": X_2[j], "label": y_2_soft[j]}

            alfa = np.random.uniform(0.1, 0.4)
            lam = np.random.beta(alfa, alfa)
            lams = np.append(lams, lam)

            mix_X, label = mix(eeg_info_1=eeg_info_1, eeg_info_2=eeg_info_2, lam=lam)

            mix_Xs = np.vstack([mix_Xs, mix_X])
            mix_ys_soft = np.vstack([mix_ys_soft, label])
            mix_ys = np.append(mix_ys, np.argmax(label))

    return mix_Xs, mix_ys, mix_ys_soft, y_1s, y_2s, lams, g1s, g2s


class StratifiedGroupKFold():

    def __init__(self, k=5, n_repeats=1, seed=None):
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
            print(y_distr, y_counts_per_group)
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

#script starts
x_path = "/Users/yeganehghamari/epilepsy-project/data/multiclass_X_new.npy"
y_path = "/Users/yeganehghamari/epilepsy-project/data/multiclass_y_new.npy"
groups_path = "/Users/yeganehghamari/epilepsy-project/data/multiclass_patients_new.npy"

verbose = True
seed = 55784899

np.random.seed(seed)
random.seed(seed)

X = np.load(x_path)
X = X.reshape(-1, np.product(X.shape[1:]))
y = np.load(y_path)
num_classes = len(np.unique(y))

patients = np.load(groups_path, allow_pickle=True)
groups = np.empty(len(patients), dtype=int)

unique_patients = np.unique(patients)
unique_groups = np.arange(len(unique_patients))
for i, patient in enumerate(unique_patients):
    groups[patients == patient] = i

best_hyperparametes = []
print('X', X.shape, 'y', y.shape, 'groups',groups.shape)

#cross validation
_, _, train_idx, test_idx = list(StratifiedGroupKFold(k=5, n_repeats=1, seed=seed).split(X, y, patients))[0]

trainX, trainy, traing = X[train_idx], y[train_idx], groups[train_idx]
testX, testy, testg = X[test_idx], y[test_idx], groups[test_idx]

#yval = ohe.transform(yval.reshape(-1,1)).toarray()
#standardize both train and test data by training on the train data
scaler = StandardScaler().fit(trainX)
trainX = scaler.transform(trainX)
testX = scaler.transform(testX)
print('trainX', np.shape(trainX), 'testX', np.shape(testX), 'trainy', np.shape(trainy), 'testy', np.shape(testy))
#save mixup set

mix_X, mix_y, mix_y_soft, y_1s, y_2s, lams, g1s, g2s = mixup(trainX, trainy, traing, size=50000)

np.save("X_mixup_train_stratified.npy", mix_X)
np.save("y_mixup_train_stratified.npy", mix_y)
np.save('y_mixup_soft_train_stratified.npy', mix_y_soft)
np.save('y_1hard.npy',y_1s)
np.save('y_2hard.npy', y_2s)
np.save('lambdas.npy',lams)
np.save('groups1.npy', g1s)
np.save('groups2.npy', g2s)

#save downsampled train set
sampler = RandomUnderSampler(sampling_strategy={2: 10000, 3: 10000, 5: 10000}, random_state=55784899)
trainX, trainy = sampler.fit_resample(trainX, trainy)
shuffler = np.random.permutation(len(trainy))
trainX, trainy, traing = trainX[shuffler], trainy[shuffler], traing[shuffler]
ohe = OneHotEncoder()
ohe.fit([[0], [1], [2], [3], [4], [5]])
trainy_soft = ohe.transform(trainy.reshape(-1, 1)).toarray()

np.save("X_orig_train_stratified_withgs.npy", trainX)
np.save("y_orig_train_stratified_withgs.npy", trainy)
np.save('ysoft_orig_train_stratified_withgs.npy', trainy_soft)
np.save('trainingpatients.npy', traing)
np.save("X_orig_test_stratified.npy", testX)
np.save("y_orig_test_stratified.npy", testy)