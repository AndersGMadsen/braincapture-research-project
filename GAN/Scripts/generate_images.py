from matplotlib import pyplot
from numpy import expand_dims, vstack
from numpy import ones
from numpy import zeros
from numpy.random import rand, randn
from numpy.random import randint
#from keras.datasets.mnist import load_data
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

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input





latent_dim=10
g_model = define_generator(latent_dim)
path = '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/generator_models/4_generator_model_weights_580.h5'
g_model.load_weights(path)

# model = load_model('generator_models/1_generator_model_200.h5')
# generate images
n_images = 20000
latent_points = generate_latent_points(10, n_images)
# generate images
X = g_model.predict(latent_points)
np.save('fake_shiv_580', X)

#save_plot(X, 300, int(np.sqrt(n_images)))