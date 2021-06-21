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
from sklearn.metrics import fbeta_score, classification_report, balanced_accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_random_state, resample

paths = {
'chew': '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/fake images/fake_chew_700.npy',
'elpp': '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/fake images/fake_elpp_700.npy',
'eyem': '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/fake images/fake_eyem_200.npy',
'musc': '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/fake images/fake_musc_400.npy',
'shiv': '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/fake images/fake_shiv_900.npy'
}

artifact_names = paths.keys()
names = ['chew', 'elpp', 'eyem', 'musc', 'shiv', 'null']

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

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

def load_GAN_data():

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
    y_soft_path = '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/Data/y_mixup_train_stratified_soft.npy'
    y_hard_path = '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/Data/y_mixup_train_stratified_hard.npy'

    X_mixup = np.load(X_path)
    y_mixup_soft = np.load(y_soft_path)
    y_mixup_hard = np.load(y_hard_path)

    return (X_mixup, y_mixup_soft, y_mixup_hard)

def run(n_real, n_fake, X_real, X_test, y_real, y_test):

    soft = False
    X_fake, y_fake = load_GAN_data()
    #(X_mixup, y_mixup_soft, y_mixup_hard) = load_mixup_data()
    #X_fake, y_fake = (X_mixup, y_mixup_soft)


    mask = resample(np.where(y_real == 5)[0], replace=False, n_samples=1043702 - 26000)
    X_real = np.delete(X_real, mask, axis=0)
    y_real = np.delete(y_real, mask, axis=0)

    if soft:
        X_fake, y_fake = (X_mixup, y_mixup_soft)

        ohe = OneHotEncoder()
        ohe.fit([[0], [1], [2], [3], [4], [5]])
        y_real = ohe.transform(y_real.reshape(-1, 1)).toarray()

    idx_real = np.random.choice(range(len(X_real)), n_real)
    idx_fake = np.random.choice(range(len(X_fake)), n_fake)

    X_train = np.concatenate((X_real[idx_real], X_fake[idx_fake]))
    y_train = np.concatenate((y_real[idx_real], y_fake[idx_fake]))

    shuffler = np.random.permutation(len(X_train))
    X_train = X_train[shuffler]
    y_train = y_train[shuffler]


    X_train = X_train.reshape(-1, 19 * 25)
    X_test = X_test.reshape(-1, 19 * 25)

    LDA = LinearDiscriminantAnalysis()
    MLP = MLPClassifier()
    MLPreg = MLPRegressor()

    model = LDA
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)

    if soft:
        y_pred = softmax(y_pred)
        y_pred = np.argmax(y_pred, axis=1)

    fbeta = fbeta_score(y_test, y_pred, average='weighted', beta=2)
    bacc = balanced_accuracy_score(y_test, y_pred)
    #report = classification_report(y_test, y_pred, target_names=names)
    return bacc



X = np.load('/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/Data/X_artifacts_only.npy')
y = np.load('/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/Data/y_artifacts_only.npy')


ns = [ [10, 10]    ,       [10, 100],   [10, 500],    [10, 1000], [10, 5000],     [10, 10000],     [10, 50000],
      [100, 10],   [100, 100], [100, 500],   [100, 1000], [100, 5000],   [100, 10000],   [100, 50000],
      [500, 10],   [500, 100],  [500, 500],  [500, 1000], [500, 5000],   [500, 10000],   [500, 50000],
      [1000, 10],  [1000, 100],  [1000, 500], [1000, 1000], [1000, 5000],  [1000, 10000],  [1000, 50000],
      [5000, 10],  [5000, 100],  [5000, 500], [5000, 1000],  [5000, 5000], [5000, 10000],  [5000, 50000],
      [10000, 10], [10000, 100], [10000, 500], [10000, 1000], [10000, 5000], [10000, 10000], [10000, 50000],
      [50000, 10], [50000, 100], [50000, 500], [50000, 1000], [50000, 5000], [50000, 10000], [50000, 50000]]

#accuracies = pd.DataFrame(index = ['0', '100', '1000', '10000', '50000'], columns = ['0', '100', '1000', '10000', '50000'])

accs = np.zeros((7, 7))

def load_data():
    X_path = '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/Data/multiclass_X_new.npy'
    y_path = '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/Data/multiclass_y_new.npy'
    groups_path = "/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/Data/multiclass_patients_new.npy"

    X = np.load(X_path).reshape(1386103, 19, 25)
    y = np.load(y_path)

    groups = np.load(groups_path, allow_pickle=True)
    unique_groups = np.unique(groups)

    _, _, train_idx, test_idx = list(StratifiedGroupKFold(k=5, n_repeats=1, seed=55784899).split(X, y, groups))[0]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    X_train = X_train.reshape(len(X_train), 19, 25, 1)
    X_test = X_test.reshape(len(X_test), 19, 25, 1)

    return (X_train, y_train), (X_test, y_test)

(X_train, y_train), (X_test, y_test) = load_data()

from tqdm import tqdm

for i, n in enumerate(ns):
    n_real, n_fake = n[0], n[1]

    scores = []

    '''for train, test in StratifiedKFold(n_splits=5, shuffle=True, random_state=None).split(X, y):
        X_train = X[train[0]: train[-1]]
        X_test = X[test[0]: test[-1]]
        y_train = y[train[0]: train[-1]]
        y_test = y[test[0]: test[-1]]

        X_train = X_train.reshape(len(X_train), 19, 25, 1)
        X_test = X_test.reshape(len(X_test), 19, 25, 1)'''

    print(n_real, n_fake)
    for j in tqdm(range(5)):
        accuracy = run(n_real, n_fake, X_train, X_test, y_train, y_test)
        scores.append(accuracy)

    acc = sum(scores) / len(scores)
    accs[i % 7][i // 7] = round(acc, 2)

#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rc('text', usetex=True)



#accs = np.load('/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/Data/test_f2_matrix_soft.npy')


font = {'family': 'serif',

        'size': '20',

        'serif': ['Computer Modern'],

        'sans-serif': ['Computer Modern']}

plt.rc('font', **font)

plt.rc('axes', titlesize=28, labelsize=22)

plt.rc('xtick', labelsize=16)

plt.rc('ytick', labelsize=16)

plt.rc('legend', fontsize=16)


labels = [10, 100, 500, 1000, 5000, 10000, 50000]

ax = sns.heatmap(accs, linewidth=0.5, robust=True,  yticklabels=labels, xticklabels=labels, annot=True, vmin=.2, vmax=.35)

plt.title(r'GAN, LDA: B. Acc.', size=28, y=1.01)
plt.xlabel('Number of Original Samples')
plt.ylabel('Number of Generated Samples')
plt.savefig("GAN LDA bacc", dpi=1000, bbox_inches = 'tight')
plt.show()


