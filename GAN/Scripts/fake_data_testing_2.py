from collections import defaultdict, Counter

import numpy as np
import random
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils import check_random_state, resample
from sklearn.metrics import fbeta_score

paths = {
'chew': '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/fake images/fake_chew_700.npy',
'elpp': '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/fake images/fake_elpp_700.npy',
'eyem': '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/fake images/fake_eyem_200.npy',
'musc': '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/fake images/fake_musc_400.npy',
'shiv': '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/fake images/fake_shiv_900.npy'
}

artifact_names = paths.keys()

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
    X_path = '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/Data/X.npy'
    y_path = '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/Data/y.npy'
    groups_path = "/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/Data/patients.npy"

    X = np.load(X_path).reshape(1386103, 19, 25)
    y = np.load(y_path)

    groups = np.load(groups_path, allow_pickle=True)
    unique_groups = np.unique(groups)

    _, _, train_idx, test_idx = list(StratifiedGroupKFold(k=5, n_repeats=1, seed=55784899).split(X, y, groups))[0]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    return (X_train, y_train), (X_test, y_test)


def load_fake_data():

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


def get_data():

    (X_train, y_train), (X_test, y_test) = load_data()
    (X_fake, y_fake) = load_fake_data()

    X_fake = np.reshape(X_fake, (100000, 19, 25))

    mask = resample(np.where(y_train == 5)[0], replace=False, n_samples=1043702 - 26000)

    X_train = np.delete(X_train, mask, axis=0)
    y_train = np.delete(y_train, mask, axis=0)

    X_train_up = X_train.copy()
    y_train_up = y_train.copy()

    for i, artifact in enumerate(range(len(artifact_names))):
        idx = np.random.choice(np.where(y_fake == i)[0], 26000 - len(X_train[y_train==i]))

        X_train_up = np.concatenate((X_train_up, X_fake[idx]))
        y_train_up = np.concatenate((y_train_up, y_fake[idx]))

    shuffler = np.random.permutation(len(X_train_up))
    X_train_up = X_train_up[shuffler]
    y_train_up = y_train_up[shuffler]

    return (X_train, y_train), (X_train_up, y_train_up), (X_test, y_test)


def run_experiment():

    (X_train, y_train), (X_train_up, y_train_up), (X_test, y_test) = get_data()

    X_train = X_train.reshape(-1, 19 * 25)
    X_train_up = X_train_up.reshape(-1, 19 * 25)
    X_test = X_test.reshape(-1, 19 * 25)

    LDA = LinearDiscriminantAnalysis()
    LDA.fit(X_train, y_train)

    LDA_balanced = LinearDiscriminantAnalysis()
    LDA_balanced.fit(X_train_up, y_train_up)

    baseline_accuracy = LDA.score(X_test, y_test)
    GAN_accuracy = LDA_balanced.score(X_test, y_test)

    baseline_pred = LDA.predict(X_test)
    GAN_pred = LDA_balanced.predict(X_test)

    baseline_fbeta = fbeta_score(y_test, baseline_pred, average='weighted', beta=2)
    GAN_fbeta = fbeta_score(y_test, GAN_pred, average='weighted', beta=2)

    return [baseline_accuracy, baseline_fbeta, GAN_accuracy, GAN_fbeta]


results = run_experiment()

print('Random under, acc.:', results[0])
print('Random under, f-beta.:', results[1])
print('Random under + GAN up, acc.:', results[2])
print('Random under + GAN up, f-beta.:', results[3])