import argparse

from matplotlib import pyplot
from numpy import expand_dims, vstack
from numpy import ones
from numpy import zeros
from numpy.random import rand, randn
from numpy.random import randint
# from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LeakyReLU

import numpy as np
from sklearn.utils import resample
from tensorflow.python.keras.layers import Reshape, Conv2DTranspose
from tensorflow.python.keras.models import load_model

from os.path import exists
from os import makedirs


parser = argparse.ArgumentParser()
parser.add_argument('--artifact', action='store', type=int, required=True)
parser.add_argument('--epochs', action='store', type=int, required=True)

args = parser.parse_args()

artifact = args.artifact
n_epochs = args.epochs

outdir = 'generator_models'


def load_data():
    x_path = "X_artifacts_only_padded.npy"
    y_path = "y_artifacts_only.npy"
    patients_path = "patients.npy"

    X = np.load(x_path).reshape(82833, 28, 28)

    y = np.load(y_path)
    print(len(y))

    y_buf = np.load("y.npy")
    patients = np.load(patients_path, allow_pickle=True)
    groups = np.empty(len(patients), dtype=int)

    unique_patients = np.unique(patients)
    unique_groups = np.arange(len(unique_patients))

    for i, patient in enumerate(unique_patients):
        groups[patients == patient] = i

    groups = groups[y_buf != 5]

    idxs = resample(np.unique(groups), replace=False, n_samples=10, random_state=1928)

    trainX = X[~np.isin(groups, idxs)]
    trainy = y[~np.isin(groups, idxs)]

    testX = X[np.isin(groups, idxs)]
    testy = y[np.isin(groups, idxs)]

    return (trainX, trainy), (testX, testy)


# define the standalone discriminator model
def define_discriminator(in_shape=(28, 28, 1)):
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
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


# define the standalone generator model
def define_generator(latent_dim):
    model = Sequential()
    # foundation for 7x7 image
    n_nodes = 128 * 7 * 7
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7, 7, 128)))
    # upsample to 14x14
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 28x28
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(1, (7, 7), activation='sigmoid', padding='same'))
    return model


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(g_model)
    # add the discriminator
    model.add(d_model)
    # compile model
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)  # loss='binary_crossentropy',
    return model


# load and prepare mnist training images
def load_real_samples(i):
    # load mnist dataset
    (trainX, trainy), (_, _) = load_data()
    # expand to 3d, e.g. add channels dimension
    X = expand_dims(trainX[trainy == i], axis=-1)
    # convert from unsigned ints to floats
    X = X.astype('float32')

    X = (X + 5.0) / 10.0

    return X


# select real samples
def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, 1))
    # save_plot(X, 0)
    return X, y


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = g_model.predict(x_input)
    # create 'fake' class labels (0)
    y = zeros((n_samples, 1))
    # save_plot(X, 0)
    return X, y


# create and save a plot of generated images
def save_plot(examples, epoch, n=5):
    # plot images
    fig, ax = pyplot.subplots(n, n)
    for i in range(n * n):
        # turn off axis
        ax[i % n, i // n].axis('off')
        # plot raw pixel data
        im = ax[i % n, i // n].imshow(examples[i, :, :, 0], cmap="magma")
        im.set_clim(0, 1)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    # save plot to file
    filename = 'chew_generated_plot_e%03d.png' % (epoch + 1)
    pyplot.savefig(filename)
    pyplot.show()


# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
    # prepare real samples
    X_real, y_real = generate_real_samples(dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real * 100, acc_fake * 100))
    # save plot
    #save_plot(x_fake, epoch)
    # save the generator model tile file
    filename = outdir + '/' + str(artifact) + '_generator_model_%03d.h5' % (epoch + 1)
    g_model.save_weights(filename)
    #g_model.save(filename)


# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=256):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # create training set for the discriminator
            X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
            # update discriminator model weights
            d_loss, _ = d_model.train_on_batch(X, y)
            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d=%.3f, g=%.3f' % (i + 1, j + 1, bat_per_epo, d_loss, g_loss))
        # evaluate the model performance, sometimes
        if (i + 1) % 100 == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim)


if not exists(outdir):
    makedirs(outdir)
    
# size of the latent space
latent_dim = 10
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)

# load image data
dataset = load_real_samples(artifact)

# train model
train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=n_epochs)
