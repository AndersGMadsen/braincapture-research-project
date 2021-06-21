from collections import defaultdict, Counter

import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.patches as mpatches

from sklearn.utils import check_random_state
import random

seed = 55784899
np.random.seed(seed)
random.seed(seed)

# SETTINGS FOR PYPLOT


plt.rc('text', usetex=True)

font = {'family': 'serif',

        'size': '20',

        'serif': ['Computer Modern'],

        'sans-serif': ['Computer Modern']}

plt.rc('font', **font)

plt.rc('axes', titlesize=28, labelsize=22)

plt.rc('xtick', labelsize=16)

plt.rc('ytick', labelsize=16)

plt.rc('legend', fontsize=16)

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

## LOADING REAL DATA
def load_data():
    X_path = '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/Data/multiclass_X_new.npy'
    y_path = '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/Data/multiclass_y_new.npy'
    groups_path = "/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/Data/multiclass_patients_new.npy"

    X = np.load(X_path).reshape(1386103, 19, 25)
    y = np.load(y_path)

    groups = np.load(groups_path, allow_pickle=True)
    unique_groups = np.unique(groups)

    _, _, train_idx, test_idx = list(StratifiedGroupKFold(k=5, n_repeats=1, seed=seed).split(X, y, groups))[0]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    return (X_train.reshape(-1, 19*25), y_train), (X_test.reshape(-1, 19*25), y_test)


(X_train, y_train), (X_test, y_test) = load_data()


# LOADING GAN IMAGES


def load_gan_data():

    fake_chew = np.load('/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/fake images/fake_chew_700.npy')
    fake_chew = 10.0 * fake_chew - 5
    fake_chew = fake_chew[:, :19, :25]

    fake_elpp = np.load('/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/fake images/fake_elpp_700.npy')
    fake_elpp = 10.0 * fake_elpp - 5
    fake_elpp = fake_elpp[:, :19, :25]

    fake_eyem = np.load('/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/fake images/fake_eyem_200.npy')
    fake_eyem = 10.0 * fake_eyem - 5
    fake_eyem = fake_eyem[:, :19, :25]

    fake_musc = np.load('/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/fake images/fake_musc_400.npy')
    fake_musc = 10.0 * fake_musc - 5
    fake_musc = fake_musc[:, :19, :25]

    fake_shiv = np.load('/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/fake images/fake_shiv_900.npy')
    fake_shiv = 10.0 * fake_shiv - 5
    fake_shiv = fake_shiv[:, :19, :25]

    fake_artifactsx = [fake_chew, fake_elpp, fake_eyem, fake_musc, fake_shiv]
    fake_artifacts = []

    for fake_artifact in fake_artifactsx:
        fake_artifacts.append(fake_artifact.reshape(len(fake_artifact), 19*25))

    return fake_artifacts


GAN_artifacts = load_gan_data()

# LOADING MIXUP IMAGES


def load_mixup_data():
    X_path = '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/Data/X_mixup_train_stratified.npy'
    y_path = '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/Data/y_mixup_train_stratified.npy'

    X_mixup = np.load(X_path)
    y_mixup = np.load(y_path)

    mixup_chew = X_mixup[y_mixup == 0]
    mixup_elpp = X_mixup[y_mixup == 1]
    mixup_eyem = X_mixup[y_mixup == 2]
    mixup_musc = X_mixup[y_mixup == 3]
    mixup_shiv = X_mixup[y_mixup == 4]
    mixup_null = X_mixup[y_mixup == 5]

    mixup_artifacts = [mixup_chew, mixup_elpp, mixup_eyem, mixup_musc, mixup_shiv]

    return mixup_artifacts, mixup_null


mixup_artifacts, mixup_null = load_mixup_data()

# ORGANIZING ARTIFACT DATA


real_chew = X_test[y_test == 0]
real_elpp = X_test[y_test == 1]
real_eyem = X_test[y_test == 2]
real_musc = X_test[y_test == 3]
real_shiv = X_test[y_test == 4]
real_null = X_test[y_test == 5]

real_artifacts = [real_chew, real_elpp, real_eyem, real_musc, real_shiv]


#%%

## ORIGINAL DATA
names = ['chew', 'elpp', 'eyem', 'musc', 'shiv']
artifact_names = ['Chewing', 'Electrode Pop', 'Eye Movement', 'Muscle', 'Shivering', 'Null']

my_colors = ["#f5cf40", "#e63f47", "#0ed280", "#fc7323", "#79218f", "#828bf2", 'b']
markers = ["*", "s", "^", "D", "P"]

pca = PCA(n_components=2)
pca.fit(X_train)

pc_X_test = pca.transform(X_test)

pc_real_artifacts = []
pc_GAN_artifacts = []
pc_mixup_artifacts = []

for real_artifact in real_artifacts:
    pc_real_artifacts.append(pca.transform(real_artifact))

for fake_artifact in GAN_artifacts:
    pc_GAN_artifacts.append(pca.transform(fake_artifact))

for fake_artifact in mixup_artifacts:
    pc_mixup_artifacts.append(pca.transform(fake_artifact))

pc_mixup_null = pca.transform(mixup_null)
pc_real_null = pca.transform(real_null)

pc1 = 0
pc2 = 1

# PC PLOT FOR REAL DATA
fig = pyplot.figure()
ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])

ax.scatter(pc_real_null[:,pc1], pc_real_null[:,pc2], c="#828bf2", label='null')

for i, pc_real_artifact in enumerate(pc_real_artifacts):
    ax.scatter(pc_real_artifact[:,pc1], pc_real_artifact[:,pc2],color=my_colors[i], marker= markers[i], label=names[i])

pyplot.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', markerscale=2)
pyplot.title(r"\textbf{PCA Plot of Original Data}", size=18, y=1.01)
pyplot.savefig("PCA Real artifacts", dpi=1000, bbox_inches = 'tight')
pyplot.show()


#%%

# PC PLOT FOR mixup DATA
fig = pyplot.figure()
ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])

ax.scatter(pc_X_test[:,pc1], pc_X_test[:,pc2], c="k", label="Test data")

idx = np.random.choice(range(len(pc_mixup_null)), 500)
ax.scatter(pc_mixup_null[:,pc1], pc_mixup_null[:,pc2], c=my_colors[5], label="null")

for i, pc_fake_artifact in enumerate(pc_mixup_artifacts):
    idx = np.random.choice(range(len(pc_fake_artifact)), 500)
    ax.scatter(pc_fake_artifact[:,pc1][idx], pc_fake_artifact[:,pc2][idx],color=my_colors[i], marker= markers[i], label=names[i])

pyplot.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', markerscale=2)
pyplot.title(r"\textbf{PCA Plot of mixup Data}", size=18, y=1.01)
pyplot.savefig("PCA mixup artifacts", dpi=1000, bbox_inches = 'tight')
pyplot.show()


# PC PLOT FOR  GAN DATA
fig = pyplot.figure()
ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])

ax.scatter(pc_X_test[:,pc1], pc_X_test[:,pc2], c="k", label="Test data")

for i, pc_fake_artifact in enumerate(pc_GAN_artifacts):
    idx = np.random.choice(range(20000), 500)
    ax.scatter(pc_fake_artifact[:,pc1][idx], pc_fake_artifact[:,pc2][idx],color=my_colors[i], marker= markers[i], label=names[i])

pyplot.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', markerscale=2)
pyplot.title(r"\textbf{PCA Plot of GAN Data}", size=18, y=1.01)
pyplot.savefig("PCA GAN artifacts", dpi=1000, bbox_inches = 'tight')
pyplot.show()

#%%
#import seaborn as sns
#sns.set_style("whitegrid")
plt.rc('axes', titlesize=18, labelsize=22)
fig, ax = pyplot.subplots(2, 3, sharey=True)
fig.tight_layout()

for i in range(5):
    idx = np.random.choice(range(20000), 500)
    ax[i % 2, i % 3].scatter(pc_real_artifacts[i][:,pc1], pc_real_artifacts[i][:,pc2],color='k')
    ax[i % 2, i % 3].scatter(pc_GAN_artifacts[i][:,pc1][idx], pc_GAN_artifacts[i][:,pc2][idx],color=my_colors[i], marker=markers[i], label=names[i])
    ax[i % 2, i % 3].set_title(artifact_names[i])


fig.suptitle(r"\textbf{PCA Plot of GAN Data on Original Data}", size=20, y=1.04)
#fig.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', markerscale=2)
fig.delaxes(ax[1, 2])
pyplot.savefig("PCA Original and GAN", dpi=1000, bbox_inches = 'tight')
pyplot.show()

plt.rc('axes', titlesize=18, labelsize=22)
fig, ax = pyplot.subplots(2, 3, sharey=True)
fig.tight_layout()

for i in range(5):
    idx = np.random.choice(range(len(pc_real_artifacts[i])), 500)
    ax[i % 2, i % 3].scatter(pc_real_artifacts[i][:,pc1], pc_real_artifacts[i][:,pc2],color='k')
    ax[i % 2, i % 3].scatter(pc_mixup_artifacts[i][:,pc1][idx], pc_mixup_artifacts[i][:,pc2][idx],color=my_colors[i], marker=markers[i], label=names[i])
    ax[i % 2, i % 3].set_title(artifact_names[i])

idx = np.random.choice(range(len(pc_mixup_null)), 500)
ax[1, 2].scatter(pc_real_null[:,pc1], pc_real_null[:,pc2], c='k')
ax[1, 2].scatter(pc_mixup_null[:,pc1], pc_mixup_null[:,pc2], c=my_colors[5], label="null")
ax[1, 2].set_title('Null')

fig.suptitle(r"\textbf{PCA Plot of mixup Data on Original Data}", size=20, y=1.04)
#fig.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', markerscale=2)
#fig.delaxes(ax[1, 2])
pyplot.savefig("PCA Original and mixup", dpi=1000, bbox_inches = 'tight')
pyplot.show()

#%%



