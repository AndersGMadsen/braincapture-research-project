from collections import defaultdict, Counter

import numpy as np
from community.community_louvain import check_random_state
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import random
from imblearn.under_sampling import RandomUnderSampler, NearMiss

from sklearn.model_selection import train_test_split
import matplotlib.patches as mpatches

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

#%%
X_path = '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/Data/multiclass_X_new.npy'
y_path = '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/Data/multiclass_y_new.npy'
groups_path = "/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/Data/patients.npy"

X = np.load(X_path)
y = np.load(y_path)

groups = np.load(groups_path, allow_pickle=True)
unique_groups = np.unique(groups)

_, _, train_idx, test_idx = list(StratifiedGroupKFold(k=5, n_repeats=1, seed=55784899).split(X, y, groups))[0]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

X_train = X_train.reshape(len(X_train), 19*25)
X_test = X_test.reshape(len(X_test), 19*25)

#%%
names = ['chew', 'elpp', 'eyem', 'musc', 'shiv', 'null']
my_colors = ["#f5cf40", "#e63f47", "#0ed280", "#fc7323", "#79218f", "#828bf2"]
markers = ["*", "s", "^", "D", "P"]

seed = 55784899
np.random.seed(seed)
idx = np.random.choice(range(len(X_test[y_test == 4])), 10)

chew = X_test[y_test == 0][idx]
elpp = X_test[y_test == 1][idx]
eyem = X_test[y_test == 2][idx]
musc = X_test[y_test == 3][idx]
shiv = X_test[y_test == 4][idx]

idx = np.random.choice(range(len(X_test[y_test == 5])), 10000)
null = X_test[y_test == 5][idx]

original_artifacts = [chew, elpp, eyem, musc, shiv]

#%%
dataX = []
datay = []

for i, artifact in enumerate(original_artifacts):
    for image in artifact:
        dataX.append(image)
        datay.append(i)

for image in null:
    dataX.append(image)
    datay.append(5)

dataX = np.array((dataX))
datay = np.array((datay))

#%%

## ORIGINAL DATA
pca = PCA(n_components=2)
pca.fit(X_train)

pc_X_test = pca.transform(X_test)


pc_original_artifacts = []

pc_null = pca.transform(null)

for artifact in original_artifacts:
    pc_original_artifacts.append(pca.transform(artifact))

pc1 = 0
pc2 = 1

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])

ax.scatter(pc_null[:,pc1], pc_null[:,pc2], c=my_colors[5], label=names[5])


for i, artifact in enumerate(pc_original_artifacts):
    ax.scatter(artifact[:,pc1], artifact[:,pc2], color=my_colors[i], marker=markers[i], label=names[i], edgecolor=(0,0,0,1))


plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', markerscale=2)
plt.ylim((-14,14))
plt.xlim((-11,17))
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(r"\textbf{PCA Plot of Original Data}", size=18, y=1.01)
plt.savefig("PCA without augmentation", dpi=1000, bbox_inches = 'tight')
plt.show()


#%%


## RANDOM UNDER SAMPLING OF NULL

rus = RandomUnderSampler(sampling_strategy={5:500}, random_state=seed)
dataX_res, datay_res = rus.fit_resample(dataX, datay)


res_chew = dataX_res[datay_res == 0]
res_elpp = dataX_res[datay_res == 1]
res_eyem = dataX_res[datay_res == 2]
res_musc = dataX_res[datay_res == 3]
res_shiv = dataX_res[datay_res == 4]
res_null = dataX_res[datay_res == 5]

res_artifacts = [res_chew, res_elpp, res_eyem, res_musc, res_shiv]
pc_res_artifacts = []

pc_null = pca.transform(res_null)

for artifact in res_artifacts:
    pc_res_artifacts.append(pca.transform(artifact))

pc1 = 0
pc2 = 1

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])

ax.scatter(pc_null[:,pc1], pc_null[:,pc2], c=my_colors[5], label=names[5])


for i, artifact in enumerate(pc_original_artifacts):
    ax.scatter(artifact[:,pc1], artifact[:,pc2], color=my_colors[i], marker=markers[i], label=names[i], edgecolor=(0,0,0,1))


plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', markerscale=2)
plt.ylim((-14,14))
plt.xlim((-11,17))
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(r"\textbf{Random Majority Down Sampling}", size=18, y=1.01)
plt.savefig("PCA random under sampling", dpi=1000, bbox_inches = 'tight')
plt.show()
plt.show()


#%%


## SMOTE

s = 500
#sampling_strategy={0: s, 1: s, 2: s, 3: s, 4: s}
smote = SMOTE(sampling_strategy={0: s, 1: s, 2: s, 3: s, 4: s}, random_state=seed)
dataX_smote, datay_smote = smote.fit_resample(dataX, datay)


smote_chew = dataX_smote[datay_smote == 0]
smote_elpp = dataX_smote[datay_smote == 1]
smote_eyem = dataX_smote[datay_smote == 2]
smote_musc = dataX_smote[datay_smote == 3]
smote_shiv = dataX_smote[datay_smote == 4]
smote_null = dataX_smote[datay_smote == 5]

smote_artifacts = [smote_chew, smote_elpp, smote_eyem, smote_musc, smote_shiv]
pc_smote_artifacts = []

pc_null = pca.transform(smote_null)

for artifact in smote_artifacts:
    pc_smote_artifacts.append(pca.transform(artifact))

pc1 = 0
pc2 = 1

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])

ax.scatter(pc_null[:,pc1], pc_null[:,pc2], c=my_colors[5], label=names[5])


for i, artifact in enumerate(pc_smote_artifacts):
    print(i)
    ax.scatter(artifact[:,pc1], artifact[:,pc2], color=my_colors[i], marker=markers[i])

for i, artifact in enumerate(pc_original_artifacts):
    ax.scatter(artifact[:,pc1], artifact[:,pc2], color=my_colors[i], marker=markers[i], label=names[i], edgecolor=(0,0,0,1))


plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', markerscale=2)
plt.ylim((-14,14))
plt.xlim((-11,17))
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(r"\textbf{Minority Up Sampling with SMOTE}", size=18, y=1.01)
plt.savefig("PCA SMOTE", dpi=1000, bbox_inches = 'tight')
plt.show()


#%%


## NEARMISS

nm = NearMiss(sampling_strategy={5: 500})
dataX_nm, datay_nm = nm.fit_resample(dataX, datay)


nm_chew = dataX_nm[datay_nm == 0]
nm_elpp = dataX_nm[datay_nm == 1]
nm_eyem = dataX_nm[datay_nm == 2]
nm_musc = dataX_nm[datay_nm == 3]
nm_shiv = dataX_nm[datay_nm == 4]
nm_null = dataX_nm[datay_nm == 5]

nm_artifacts = [nm_chew,nm_elpp, nm_eyem, nm_musc,nm_shiv]
pc_nm_artifacts = []

pc_null = pca.transform(nm_null)

for artifact in nm_artifacts:
    pc_nm_artifacts.append(pca.transform(artifact))

pc1 = 0
pc2 = 1

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])

ax.scatter(pc_null[:,pc1], pc_null[:,pc2], c=my_colors[5], label=names[5])


for i, artifact in enumerate(pc_nm_artifacts):
    ax.scatter(artifact[:,pc1], artifact[:,pc2], color=my_colors[i], marker=markers[i], label=names[i], edgecolor=(0,0,0,1))


plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', markerscale=2)
plt.ylim((-14,14))
plt.xlim((-11,17))
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(r"\textbf{Majority Down Sampling with NearMiss}", size=18, y=1.01)
plt.savefig("PCA NearMiss", dpi=1000, bbox_inches = 'tight')
plt.show()