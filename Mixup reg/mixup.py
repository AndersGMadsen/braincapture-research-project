import numpy as np

def mix(eeg_info_1=None, eeg_info_2=None, lambd=None):

    X1, X2 = eeg_info_1["eeg"], eeg_info_2["eeg"]

    y1, y2 = eeg_info_1["label"], eeg_info_2["label"]

    mix_X = X1 * lambd + X2 * (1. - lambd)

    label = y1 * lambd + y2 * (1. - lambd)

    return mix_X, label


def mixupreg(X_train=None, y_train=None, size=None, alpha=None):
    # Number of mixup augmentation examples - default is
    # N = int(len(y_train) / 10) if not size or size > len(y_train) else size
    N = size
    N2 = int(N / 2)
    print(N2)
    #print(X_train.shape, y_train.shape)
    # If method is 'split' we need 10 extra examples per mixed example, since we will generate 10 new
    # examples for each mixup data point

    mix_Xs = np.zeros((N2, np.size(X_train, 1)))
    mix_ys = np.zeros((N2, np.size(y_train, 1)))

    # choose mixup batch randomly from the training data
    # indices = np.random.permutation(len(X_train))[:N * 2]

    # N2 = int(len(indices) / 2) if N >= np.size(y_train, 0) / 2 else N
    X_1, X_2 = X_train[:N2], X_train[-N:] #allow for #fake==#real, doesnt guarantee that the data in X1 and X2 are not the same
    y_1, y_2 = y_train[:N2], y_train[-N:]
    #X_1, X_2 = X_train[:N2], X_train[N2:2 * N2] #if want to make sure that the data you choose to do mixup on is not the same
    #y_1, y_2 = y_train[:N2], y_train[N2:2 * N2]
    print(y_1.shape, y_2.shape)
    # X_1, X_2 = X_train[indices[:N2]], X_train[indices[N2:]]
    # y_1, y_2 = y_train[indices[:N2]], y_train[indices[N2:]]
    # print(y_train.shape(), y_1.shape())

    for i in range(N2-1):
        eeg_info_1, eeg_info_2 = {"eeg": X_1[i], "label": y_1[i]}, {"eeg": X_2[i], "label": y_2[i]}
        # lambd = np.random.uniform(0.1, 0.4)  # must be beta
        lambd = np.random.beta(alpha, alpha)
        #print(lambd)
        mix_X, mix_y = mix(eeg_info_1=eeg_info_1, eeg_info_2=eeg_info_2, lambd=lambd)
        mix_Xs[i] = mix_X
        mix_ys[i] = mix_y
    # print(mix_ys)

    return mix_Xs, mix_ys


def converttohardlabel(y_pred_mix, method=''):
    y_pred_mix_hard = np.zeros((np.size(y_pred_mix, 0), np.size(y_pred_mix, 1)))

    if method == 'Maximum likelihood':
        for i in range(len(y_pred_mix)):  # loop over rows

            if np.argmax(y_pred_mix, axis=1)[i] == 5:
                y_pred_mix_hard[i] = [0, 0, 0, 0, 0, 1]

            elif np.argmax(y_pred_mix, axis=1)[i] == 4:
                y_pred_mix_hard[i] = [0, 0, 0, 0, 1, 0]

            elif np.argmax(y_pred_mix, axis=1)[i] == 3:
                y_pred_mix_hard[i] = [0, 0, 0, 1, 0, 0]

            elif np.argmax(y_pred_mix, axis=1)[i] == 2:
                y_pred_mix_hard[i] = [0, 0, 1, 0, 0, 0]

            elif np.argmax(y_pred_mix, axis=1)[i] == 1:
                y_pred_mix_hard[i] = [0, 1, 0, 0, 0, 0]

            elif np.argmax(y_pred_mix, axis=1)[i] == 0:
                y_pred_mix_hard[i] = [1, 0, 0, 0, 0, 0]

            else:
                print(y_pred_mix[i])

    return y_pred_mix_hard

