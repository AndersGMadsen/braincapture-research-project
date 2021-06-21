from collections import defaultdict, Counter
import numpy as np
from sklearn.decomposition import PCA
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt



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
X_test = X_test.reshape(-1, 19*25)


def load_mixup_data():
    X_path = '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/Data/X_mixup_train_stratified.npy'
    y_soft_path = '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/Data/y_mixup_train_stratified_soft.npy'
    y_hard_path = '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/Data/y_mixup_train_stratified_hard.npy'

    X_mixup = np.load(X_path)
    y_mixup_soft = np.load(y_soft_path)
    y_mixup_hard = np.load(y_hard_path)

    return (X_mixup, y_mixup_soft, y_mixup_hard)


(xfake, yfake, y_mixup_hard) = load_mixup_data()

yfakemax = yfake.argsort()[:, -2:][:, ::-1]
Ymax = yfakemax[:, 0]
Ymin = yfakemax[:, 1]

n = 10000
fake_chew = xfake[Ymax == 0]
fake_chew_ymin = Ymin[Ymax == 0]
fake_chew_ymax = Ymax[Ymax == 0]

idx = np.random.choice(range(len(fake_chew)), n)
fake_chew, fake_chew_ymin, fake_chew_ymax = fake_chew[idx], fake_chew_ymin[idx], fake_chew_ymax[idx]

fake_elpp = xfake[Ymax == 1]
fake_elpp_ymin = Ymin[Ymax == 1]
fake_elpp_ymax = Ymax[Ymax == 1]
# fake_elpp, fake_elpp_ymin = shuffle(fake_elpp, fake_elpp_ymin)[:n]

idx = np.random.choice(range(len(fake_elpp)), n)
fake_elpp, fake_elpp_ymin, fake_elpp_ymax = fake_elpp[idx], fake_elpp_ymin[idx], fake_elpp_ymax[idx]

fake_eyem = xfake[Ymax == 2]
fake_eyem_ymin = Ymin[Ymax == 2]
fake_eyem_ymax = Ymax[Ymax == 2]

# fake_eyem, fake_eyem_ymin = shuffle(fake_eyem, fake_eyem_ymin)[:n]
fake_eyem, fake_eyem_ymin, fake_eyem_ymax = fake_eyem[idx], fake_eyem_ymin[idx], fake_eyem_ymax[idx]

fake_musc = xfake[Ymax == 3]
fake_musc_ymin = Ymin[Ymax == 3]
fake_musc_ymax = Ymax[Ymax == 3]
# fake_musc, fake_musc_ymin = shuffle(fake_musc, fake_musc_ymin)[:n]
idx = np.random.choice(range(len(fake_musc)), n)
fake_musc, fake_musc_ymin, fake_musc_ymax = fake_musc[idx], fake_musc_ymin[idx], fake_musc_ymax[idx]

fake_shiv = xfake[Ymax == 4]
fake_shiv_ymin = Ymin[Ymax == 4]
fake_shiv_ymax = Ymax[Ymax == 4]
# fake_shiv, fake_shiv_ymin = shuffle(fake_shiv, fake_shiv_ymin)[:n]
idx = np.random.choice(range(len(fake_shiv)), n)
fake_shiv, fake_shiv_ymin, fake_shiv_ymax = fake_shiv[idx], fake_shiv_ymin[idx], fake_shiv_ymax[idx]

fake_null = xfake[Ymax == 5]
fake_null_ymin = Ymin[Ymax == 5]
fake_null_ymax = Ymax[Ymax == 5]
# fake_null, fake_null_ymin = shuffle(fake_null, fake_null_ymin)[:n]
idx = np.random.choice(range(len(fake_null)), n)
fake_null, fake_null_ymin, fake_null_ymax = fake_null[idx], fake_null_ymin[idx], fake_null_ymax[idx]

fake_artifacts = [fake_null, fake_chew, fake_elpp, fake_eyem, fake_musc, fake_shiv]
fake_artifacts_ymin = [fake_null_ymin, fake_chew_ymin, fake_elpp_ymin, fake_eyem_ymin, fake_musc_ymin, fake_shiv_ymin]
fake_artifacts_ymax = [fake_null_ymax, fake_chew_ymax, fake_elpp_ymax, fake_eyem_ymax, fake_musc_ymax, fake_shiv_ymax]

# start PCA
names = ['null', 'chew', 'elpp', 'eyem', 'musc', 'shiv']
artifact_names = ['Null', 'Chewing', 'Electrode Pop', 'Eye Movement', 'Muscle', 'Shivering']

# my_colors = ['yellow', 'red', 'lime', 'orange', 'purple']

# my_colors = ["#f7da65", "#ec656c", "#3fdc99", "#fd8f50", "#ab74b8", "#828bf2"]
my_colors = ["#828bf2", "#f5cf40", "#e63f47", "#0ed280", "#fc7323", "#79218f"]

pca = PCA(n_components=2)
# pca = PCA(n_components=5)
pca.fit(X_train.reshape(-1, 19*25))

pc_X_test = pca.transform(X_test.reshape(-1, 19*25))

pc_real_artifacts = []
pc_fake_artifacts = []

for fake_artifact in fake_artifacts:
    pc_fake_artifacts.append(pca.transform(fake_artifact))

pc1 = 0
pc2 = 1

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])

ax.scatter(pc_X_test[:, pc1], pc_X_test[:, pc2], c="black")
for i, pc_fake_artifact in enumerate(pc_fake_artifacts):

    for j in range(6):
        if i != j:
            if len(pc_fake_artifact[fake_artifacts_ymin[i] == j]) > 0:
                idx = np.random.choice(range(len(pc_fake_artifact[fake_artifacts_ymin[i] == j])), 20)
                new_pc_fake_artifact = pc_fake_artifact[fake_artifacts_ymin[i] == j][idx]
            ax.plot(new_pc_fake_artifact[:, pc1], new_pc_fake_artifact[:, pc2], c=my_colors[i],
                    marker='.', linestyle='',
                    markersize=15,
                    markeredgewidth=0,
                    markerfacecoloralt=my_colors[j],
                    # markerfacecoloralt='green',
                    markeredgecolor='white',
                    fillstyle='left')
#%%


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
pca.fit(X_train.reshape(-1, 19*25))


pc_real_artifacts = []
pc_GAN_artifacts = []
pc_mixup_artifacts = []

for real_artifact in real_artifacts:
    pc_real_artifacts.append(pca.transform(real_artifact))

pc_real_null = pca.transform(real_null)

pc1 = 0
pc2 = 1
ax.scatter(pc_real_null[:,pc1][0], pc_real_null[:,pc2][0], c=my_colors[5], label=names[0])
for i, pc_real_artifact in enumerate(pc_real_artifacts):
    ax.scatter(pc_real_artifact[:,pc1][0], pc_real_artifact[:,pc2][0],color=my_colors[i], label=names[i])

plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', markerscale=2.5)

#plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', markerscale=2)
plt.title(r"\textbf{PCA Plot of Soft mixup Data}", size=18)
plt.savefig("PCA soft mixup data", dpi=1000, bbox_inches = 'tight')
plt.show()
