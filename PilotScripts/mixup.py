import numpy as np

# Mixing two training data points and returning the mixed data point and its label between 0 and 1
def mix(eeg_info_1=None, eeg_info_2=None, lambd=None):

    X1, X2 = eeg_info_1["eeg"], eeg_info_2["eeg"]

    y1, y2 = eeg_info_1["label"], eeg_info_2["label"]

    mix_X = X1 * lambd + X2 * (1. - lambd)

    label = y1 * lambd + y2 * (1. - lambd)

    return mix_X, label


# Takes a training set (and labels) as input and returns a modifed training set, where the mixup augmentations have been
# added
def mixup(X_train=None, y_train=None, size=None, method='split'):
    # Number of mixup augmentation examples - default is
    N = int(len(y_train)/10) if not size or size > len(y_train) else size

    # If method is 'split' we need 10 extra examples per mixed example, since we will generate 10 new
    # examples for each mixup data point
    factor = 10 if method == 'split' else 1

    mix_Xs = np.zeros((N * factor, len(X_train[0])))
    mix_ys = np.zeros(N * factor)

    # choose mixup batch randomly from the training data
    indices = np.random.permutation(len(X_train))[:N * 2 * factor]

    N2 = int(len(indices)/2) if N >= len(y_train)/2 else N

    X_1, X_2 = X_train[indices[:N2]], X_train[indices[N2:]]
    y_1, y_2 = y_train[indices[:N2]], y_train[indices[N2:]]

    for i in range(len(y_1)):

        eeg_info_1, eeg_info_2 = {"eeg": X_1[i], "label": y_1[i]}, {"eeg": X_2[i], "label": y_2[i]}
        lambd = np.random.uniform(0, 1)
        mix_X, label = mix(eeg_info_1=eeg_info_1, eeg_info_2=eeg_info_2, lambd=lambd)

        if method == 'split':
            # Add 10 identical instances of each mixup augmented data point
            for j in range(10):
                mix_Xs[10 * i + j, :] = mix_X
                mix_ys[10 * i + j] = eeg_info_1['label'] if j + 1 <= int(round(lambd, 0))*10 else eeg_info_2['label']

        # Assign label to the original label which were weighted higher in the generation og the new X point
        elif method == 'largest':
            mix_Xs[i] = mix_X
            mix_ys[i] = eeg_info_1['label'] if lambd >= 0.5 else eeg_info_2['label']

        # Sample the final label where each probability for a label corresponds to how much it has been weighted in the
        # mixup
        elif method == 'sample':
            mix_ys[i] = np.random.choice([eeg_info_1["label"], eeg_info_2["label"]], 1, p=[lambd, 1.-lambd])

    # Add mixups to training set
    X_train_mixup, y_train_mixup = np.concatenate([mix_Xs, X_train]), np.concatenate([mix_ys, y_train])

    # Shuffle the data and the new mixup augmentations before returning
    shuffler = np.random.permutation(len(X_train_mixup))
    X_train_mixup, y_train_mixup = X_train_mixup[shuffler], y_train_mixup[shuffler]

    return X_train_mixup, y_train_mixup
