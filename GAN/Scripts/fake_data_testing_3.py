#from keras.datasets.mnist import load_data
from collections import defaultdict, Counter

import pandas as pd
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LeakyReLU

import numpy as np
from sklearn.metrics import fbeta_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils import check_random_state

paths = {
'chew': '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/fake images/fake_chew_700.npy',
'elpp': '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/fake images/fake_elpp_700.npy',
'eyem': '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/fake images/fake_eyem_200.npy',
'musc': '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/fake images/fake_musc_400.npy',
'shiv': '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/fake images/fake_shiv_900.npy'
}

artifact_names = paths.keys()

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

# define the standalone discriminator model
def define_discriminator(in_shape=(19, 25, 1)):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def get_fake_images():

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


def run(n_real, n_fake, X_real, X_test, y_real, y_test):

    X_fake, y_fake = get_fake_images()

    idx_real = np.random.choice(range(len(X_real)), n_real)
    idx_fake = np.random.choice(range(len(X_fake)), n_fake)

    X_train = np.concatenate((X_real[idx_real], X_fake[idx_fake]))
    y_train = np.concatenate((y_real[idx_real], y_fake[idx_fake]))

    shuffler = np.random.permutation(len(X_train))
    X_train = X_train[shuffler]
    y_train = y_train[shuffler]

    '''d_model = define_discriminator()
    d_model.fit(X_train, y_train, epochs=10, verbose=False)'''

    X_train = X_train.reshape(-1, 19 * 25)
    X_test = X_test.reshape(-1, 19 * 25)

    LDA = LinearDiscriminantAnalysis()
    LDA.fit(X_train, y_train)

    y_pred = LDA.predict(X_test)

    fbeta = fbeta_score(y_test, y_pred, average='weighted', beta=2)

    return fbeta



X = np.load('/Data/X_artifacts_only.npy')
y = np.load('/Data/y_artifacts_only.npy')


ns = [ [10, 10]    ,       [0, 100],   [0, 500],    [0, 1000], [0, 5000],     [0, 10000],     [0, 50000],
      [100, 0],   [100, 100], [100, 500],   [100, 1000], [100, 5000],   [100, 10000],   [100, 50000],
      [500, 0],   [500, 100],  [500, 500],  [500, 1000], [500, 5000],   [500, 10000],   [500, 50000],
      [1000, 0],  [1000, 100],  [1000, 500], [1000, 1000], [1000, 5000],  [1000, 10000],  [1000, 50000],
      [5000, 0],  [5000, 100],  [5000, 500], [5000, 1000],  [5000, 5000], [5000, 10000],  [5000, 50000],
      [10000, 0], [10000, 100], [10000, 500], [10000, 1000], [10000, 5000], [10000, 10000], [10000, 50000],
      [50000, 0], [50000, 100], [50000, 500], [50000, 1000], [50000, 5000], [50000, 10000], [50000, 50000]]

#accuracies = pd.DataFrame(index = ['0', '100', '1000', '10000', '50000'], columns = ['0', '100', '1000', '10000', '50000'])

accs = np.zeros((7, 7))

for i, n in enumerate(ns):
    n_real, n_fake = n[0], n[1]

    scores = []

    for train, test in StratifiedKFold(n_splits=5, shuffle=True, random_state=None).split(X, y):
        X_train = X[train[0]: train[-1]]
        X_test = X[test[0]: test[-1]]
        y_train = y[train[0]: train[-1]]
        y_test = y[test[0]: test[-1]]

        X_train = X_train.reshape(len(X_train), 19, 25, 1)
        X_test = X_test.reshape(len(X_test), 19, 25, 1)

        accuracy = run(n_real, n_fake, X_train, X_test, y_train, y_test)
        scores.append(accuracy)

        accuracy = sum(scores) / len(scores)

    accs[i%7][i//7] = accuracy


    print()
    print('n_real:', n_real, ', n_fake:', n_fake, ', score:', accuracy)
    #if n_real != 2:
     #   accuracies[str(n_real)][str(n_fake)] = accuracy


#%%
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('text', usetex=True)

accs[0][1] = 0.34

font = {'family': 'serif',

        'size': '20',

        'serif': ['Computer Modern'],

        'sans-serif': ['Computer Modern']}

plt.rc('font', **font)

plt.rc('axes', titlesize=28, labelsize=22)

plt.rc('xtick', labelsize=16)

plt.rc('ytick', labelsize=16)

plt.rc('legend', fontsize=16)


labels = [0, 100, 500, 1000, 5000, 10000, 50000]


import seaborn as sns
ax = sns.heatmap(accs, linewidth=0.5, yticklabels=labels, xticklabels=labels, annot=True)
plt.title(r'f2 Score')
plt.xlabel('Number of Original Images')
plt.ylabel('Number of Generated Images')
plt.savefig("GAN matrix", dpi=1000, bbox_inches = 'tight')
plt.show()


