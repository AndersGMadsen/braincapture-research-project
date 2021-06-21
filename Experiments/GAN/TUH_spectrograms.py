
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from matplotlib import pyplot


from keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm


# Fonts for pyplot
plt.rcParams['font.sans-serif'] = "Georgia"
plt.rcParams['font.family'] = "sans-serif"

SMALL_SIZE = 11
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

X = np.load('/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/Data/X_artifacts_only.npy')
y = np.load('/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/Data/y_artifacts_only.npy')



patients_path = '/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/Data/patients.npy'

X = X.reshape(-1, np.product(X.shape[1:]))

#%%

num_classes = len(np.unique(y))

patients = np.load(patients_path, allow_pickle=True)
groups = np.empty(len(patients), dtype=int)

unique_patients = np.unique(patients)
unique_groups = np.arange(len(unique_patients))

for i, patient in enumerate(unique_patients):
    groups[patients == patient] = i

label_dict = {'chew': 0, 'elpp': 1, 'eyem': 2, 'musc': 3, 'shiv': 4, 'null': 5}



'''X_pad = []

for i, image in tqdm(enumerate(X)):
        new_image = np.pad(image.reshape(19, 25), ((0, 9), (0, 3)), 'constant', constant_values=(0))
        X_pad.append(new_image.flatten())

X_pad = np.array(X_pad)
np.save('X_artifacts_only_padded', X_pad)
print('saved')'''

#%%

chew, elpp, eyem, musc, shiv, null = X[y == 0], X[y == 1], X[y == 2], X[y == 3], X[y == 4], X[y == 5]
artifacts = [chew, elpp, eyem, musc, shiv]
artifact_names = ['Chewing', 'Electrode Pop', 'Eye Movement', 'Muscle', 'Shivering', 'Null']

#%%

## Random spectrograms

for i, artifact in enumerate(artifacts):
    fig, ax = plt.subplots(3, 3)

    for row in range(3):
        for col in range(3):
            idx = np.random.choice(len(artifact))
            x = np.reshape(artifact[idx], (19, 25))
            im = ax[row, col].imshow(x, cmap="magma")
            im.set_clim(-2,2)

    for a in ax.flat:
        a.label_outer()
    fig.suptitle(artifact_names[i])

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    fig.show()


#%%


## Average spectrograms

plt.rc('axes', titlesize=14)
plt.rc('ytick', labelsize=8)


fig, ax = plt.subplots(2, 3, sharey=True)
fig.tight_layout()

electrode_names = ['Fp1', 'F7', 'T3', 'T5', 'F3', 'C3', 'P3', 'O1', 'Cz', 'Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6', 'A1', 'A2']

for i, artifact in enumerate(artifacts):
    x = (sum(artifact) / len(artifact)).reshape(19, 25)

    im = ax[i % 2, i % 3].imshow(x, cmap="magma")
    im.set_clim(-1, 1)
    ax[i % 2, i % 3].set_title(artifact_names[i])

    if i == 5:
        ax[i % 2, i % 3].set_visible(False)

fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
fig.colorbar(im, cax=cbar_ax)

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.1)


fig.show()

#%%
for i, artifact in enumerate(artifacts):
    print(artifact_names[i], np.sum(artifact))