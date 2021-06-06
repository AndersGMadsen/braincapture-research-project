#from keras.datasets.mnist import load_data
import pandas as pd
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LeakyReLU

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

paths = {
'chew': '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/fake images/fake_chew_300.npy',
'elpp': '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/fake images/fake_elpp_200.npy',
'eyem': '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/fake images/fake_eyem_110.npy',
'musc': '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/fake images/fake_musc_170.npy',
'shiv': '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/fake images/fake_shiv_580.npy'
}

artifact_names = paths.keys()


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

    X_train = np.concatenate((X_real[:n_real], X_fake[:n_fake]))
    y_train = np.concatenate((y_real[:n_real], y_fake[:n_fake]))

    shuffler = np.random.permutation(len(X_train))
    X_train = X_train[shuffler]
    y_train = y_train[shuffler]

    '''d_model = define_discriminator()
    d_model.fit(X_train, y_train, epochs=10, verbose=False)'''

    X_train = X_train.reshape(-1, 19 * 25)
    X_test = X_test.reshape(-1, 19 * 25)

    LDA = LinearDiscriminantAnalysis()
    LDA.fit(X_train, y_train)

    return LDA.score(X_test, y_test)



X = np.load('/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/Data/X_artifacts_only.npy')
y = np.load('/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/Data/y_artifacts_only.npy')



ns = [[0, 100], [0, 1000], [0, 10000], [0, 50000],
      [100, 0], [100, 100], [100, 1000], [100, 10000], [100, 50000],
      [1000, 0], [1000, 100], [1000, 1000], [1000, 10000], [1000, 50000],
      [10000, 0], [10000, 100], [10000, 1000], [10000, 10000], [10000, 50000],
      [50000, 0], [50000, 100], [50000, 1000], [50000, 10000], [50000, 50000]]

accuracies = pd.DataFrame(index = ['0', '100', '1000', '10000', '50000'], columns = ['0', '100', '1000', '10000', '50000'])

for n in ns:
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

    print()
    print('n_real:', n_real, ', n_fake:', n_fake, ', score:', accuracy)
    if n_real != 2:
        accuracies[str(n_real)][str(n_fake)] = accuracy

