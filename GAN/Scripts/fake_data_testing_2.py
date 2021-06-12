from collections import defaultdict, Counter

import numpy as np
import random

from imblearn.over_sampling import SMOTE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils import check_random_state, resample
from sklearn.metrics import fbeta_score, balanced_accuracy_score, classification_report


artifact_names = ['chew', 'elpp', 'eyem', 'musc', 'shiv']
names = ['chew', 'elpp', 'eyem', 'musc', 'shiv', 'null']

seed = 55784899
np.random.seed(seed)
random.seed(seed)


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


def load_data():
    X_path = '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/Data/multiclass_X_new.npy'
    y_path = '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/Data/multiclass_y_new.npy'
    groups_path = "/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/Data/patients.npy"

    X = np.load(X_path).reshape(1386103, 19, 25)
    y = np.load(y_path)

    groups = np.load(groups_path, allow_pickle=True)
    unique_groups = np.unique(groups)

    _, _, train_idx, test_idx = list(StratifiedGroupKFold(k=5, n_repeats=1, seed=55784899).split(X, y, groups))[0]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    return (X_train, y_train), (X_test, y_test)


def load_GAN_data():
    paths = {
        'chew': '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/fake images/fake_chew_700.npy',
        'elpp': '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/fake images/fake_elpp_700.npy',
        'eyem': '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/fake images/fake_eyem_200.npy',
        'musc': '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/fake images/fake_musc_400.npy',
        'shiv': '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/fake images/fake_shiv_900.npy'
    }

    artifact_names = paths.keys()

    X_fake = []
    y_fake = []

    for i, artifact in enumerate(artifact_names):

        temp_artifact = np.load(paths[artifact])

        temp_artifact = 10.0 * temp_artifact - 5
        temp_artifact = temp_artifact [:, :19, :25]

        for image in temp_artifact:
            X_fake.append(image)

        for image in range(len(temp_artifact)):
            y_fake.append(i)

    # shuffle data
    shuffler = np.random.permutation(len(X_fake))
    X_fake = np.array(X_fake)[shuffler.astype(int)]
    y_fake = np.array(y_fake)[shuffler.astype(int)]

    return X_fake, y_fake


def load_mixup_data():
    X_path = '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/Data/X_mixup_train_stratified.npy'
    y_path = '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/Data/y_mixup_train_stratified.npy'

    X_mixup = np.load(X_path)
    y_mixup = np.load(y_path)

    return (X_mixup, y_mixup)


def get_data():

    (X_train, y_train), (X_test, y_test) = load_data()
    (X_GAN, y_GAN) = load_GAN_data()
    (X_mixup, y_mixup) = load_mixup_data()

    X_GAN = np.reshape(X_GAN, (100000, 19, 25))
    X_mixup = np.reshape(X_mixup, (53050, 19, 25))

    n_under = 30000

    mask = resample(np.where(y_train == 5)[0], replace=False, n_samples=1043702 - 50000)

    X_train = np.delete(X_train, mask, axis=0)
    y_train = np.delete(y_train, mask, axis=0)

    X_train_SMOTE = X_train.copy()
    y_train_SMOTE = y_train.copy()

    X_train_GAN = X_train.copy()
    y_train_GAN = y_train.copy()

    X_train_mixup = X_train.copy()
    y_train_mixup = y_train.copy()

    s = n_under
    smote = SMOTE(sampling_strategy={0: s, 1: s, 2: s, 3: s, 4: s}, random_state=seed)
    X_train_SMOTE, y_train_SMOTE = smote.fit_resample(X_train_SMOTE.reshape(-1, 19*25), y_train_SMOTE)

    shuffler_SMOTE = np.random.permutation(len(X_train_SMOTE))
    X_train_SMOTE = X_train_SMOTE[shuffler_SMOTE]
    y_train_SMOTE = y_train_SMOTE[shuffler_SMOTE]

    for i, artifact in enumerate(range(len(artifact_names))):
        idx_GAN = np.random.choice(np.where(y_GAN == i)[0], n_under - len(X_train[y_train == i]))
        idx_mixup = np.random.choice(np.where(y_mixup == i)[0], n_under - len(X_train[y_train == i]))

        X_train_GAN = np.concatenate((X_train_GAN, X_GAN[idx_GAN]))
        y_train_GAN = np.concatenate((y_train_GAN, y_GAN[idx_GAN]))

        X_train_mixup = np.concatenate((X_train_mixup, X_mixup[idx_mixup]))
        y_train_mixup = np.concatenate((y_train_mixup, y_mixup[idx_mixup]))

    shuffler_GAN = np.random.permutation(len(X_train_GAN))
    X_train_GAN = X_train_GAN[shuffler_GAN]
    y_train_GAN = y_train_GAN[shuffler_GAN]

    shuffler_mixup = np.random.permutation(len(X_train_mixup))
    X_train_mixup = X_train_mixup[shuffler_mixup]
    y_train_mixup = y_train_mixup[shuffler_mixup]

    return (X_train, y_train), (X_train_SMOTE, y_train_SMOTE), (X_train_GAN, y_train_GAN), (X_train_mixup, y_train_mixup), (X_test, y_test)


def run_experiment():

    (X_train, y_train), (X_train_SMOTE, y_train_SMOTE), (X_train_GAN, y_train_GAN), (X_train_mixup, y_train_mixup), (X_test, y_test) = get_data()

    X_train = X_train.reshape(-1, 19 * 25)
    X_train_SMOTE = X_train_SMOTE.reshape(-1, 19 * 25)
    X_train_GAN = X_train_GAN.reshape(-1, 19 * 25)
    X_train_mixup = X_train_mixup.reshape(-1, 19 * 25)
    X_test = X_test.reshape(-1, 19 * 25)

    LDA = LinearDiscriminantAnalysis()
    LDA.fit(X_train, y_train)

    LDA_SMOTE = LinearDiscriminantAnalysis()
    LDA_SMOTE.fit(X_train_SMOTE, y_train_SMOTE)

    LDA_GAN = LinearDiscriminantAnalysis()
    LDA_GAN.fit(X_train_GAN, y_train_GAN)

    LDA_mixup = LinearDiscriminantAnalysis()
    LDA_mixup.fit(X_train_mixup, y_train_mixup)

    baseline_pred = LDA.predict(X_test)
    SMOTE_pred = LDA_SMOTE.predict(X_test)
    GAN_pred = LDA_GAN.predict(X_test)
    mixup_pred = LDA_mixup.predict(X_test)

    baseline_bacc = balanced_accuracy_score(y_test, baseline_pred)
    SMOTE_bacc = balanced_accuracy_score(y_test, SMOTE_pred)
    GAN_bacc = balanced_accuracy_score(y_test, GAN_pred)
    mixup_bacc = balanced_accuracy_score(y_test, mixup_pred)

    baseline_f1 = fbeta_score(y_test, baseline_pred, average='weighted', beta=1)
    SMOTE_f1 = fbeta_score(y_test, SMOTE_pred, average='weighted', beta=1)
    GAN_f1 = fbeta_score(y_test, GAN_pred, average='weighted', beta=1)
    mixup_f1 = fbeta_score(y_test, mixup_pred, average='weighted', beta=1)

    baseline_f2 = fbeta_score(y_test, baseline_pred, average='weighted', beta=2)
    SMOTE_f2 = fbeta_score(y_test, SMOTE_pred, average='weighted', beta=2)
    GAN_f2 = fbeta_score(y_test, GAN_pred, average='weighted', beta=2)
    mixup_f2 = fbeta_score(y_test, mixup_pred, average='weighted', beta=2)

    baseline_report = classification_report(y_test, baseline_pred, target_names=names)
    SMOTE_report = classification_report(y_test, SMOTE_pred, target_names=names)
    GAN_report = classification_report(y_test, GAN_pred, target_names=names)
    mixup_report = classification_report(y_test, mixup_pred, target_names=names)

    results = [baseline_bacc, baseline_f1, baseline_f2, baseline_report,
               SMOTE_bacc, SMOTE_f1, SMOTE_f2, SMOTE_report,
               GAN_bacc, GAN_f1, GAN_f2, GAN_report,
               mixup_bacc, mixup_f1, mixup_f2, mixup_report]

    return results


results = run_experiment()

#%%
print('Random under, b. acc.:', results[0])
print('Random under, f1:', results[1])
print('Random under, f2:', results[2])
print('Random under, report:')
print(results[3])
print()
print('Random under + SMOTE, b. acc.:', results[4])
print('Random under + SMOTE, f1:', results[5])
print('Random under + SMOTE, f2:', results[6])
print('Random under + SMOTE, report:')
print(results[7])
print()
print('Random under + GAN, b. acc.:', results[8])
print('Random under + GAN, f1:', results[9])
print('Random under + GAN, f2:', results[10])
print('Random under + GAN, report:')
print(results[11])
print()
print('Random under + mixup, b. acc.:', results[12])
print('Random under + mixup, f1:', results[13])
print('Random under + mixup, f2:', results[14])
print('Random under + mixup, report:')
print(results[15])