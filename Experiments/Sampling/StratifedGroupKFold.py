import numpy as np
import random
from collections import Counter, defaultdict
from tqdm import tqdm
from sklearn.utils import check_random_state
# warnings.simplefilter("ignore", UserWarning)

x_path = r"C:\Users\andersgm\Documents\Courses\02466 Project work F21\Project\multiclass_X_new.npy"
y_path = r"C:\Users\andersgm\Documents\Courses\02466 Project work F21\Project\multiclass_y_new.npy"
groups_path = r"C:\Users\andersgm\Documents\Courses\02466 Project work F21\Project\multiclass_patients_new.npy"
verbose = True
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
                
                np.random.shuffle(train_indices)
                np.random.shuffle(test_indices)
                
                # Yields the indices as they are needed
                yield repeat, fold, train_indices, test_indices


# %%

X = np.load(x_path)
X = X.reshape(-1, np.product(X.shape[1:]))
y = np.load(y_path)
num_classes = len(np.unique(y))

groups = np.load(groups_path, allow_pickle=True)
unique_groups = np.unique(groups)

# %%
n_outer = 5
n_repeats = 500
outerfold = StratifiedGroupKFold(k=n_outer, n_repeats=n_repeats, seed=seed)

par_seen = []
val_seen = []


for repeat, fold, par_idxs, val_idxs in tqdm(outerfold.split(X, y, groups), total=n_repeats*n_outer):
    # -----------------------------------------------------------------------
    par_seen.append(np.sort(par_idxs))
    val_seen.append(np.sort(val_idxs))
    #-----------------------------------------------------------------------

#%%
for i in range(n_outer*n_repeats):
    for j in range(i+1, n_outer*n_repeats):
        if len(par_seen[i]) == len(par_seen[j]) and np.all(par_seen[i] == par_seen[j]):
            print(False)

        if len(val_seen[i]) == len(val_seen[j]) and np.all(val_seen[i] == val_seen[j]):
            print(False)