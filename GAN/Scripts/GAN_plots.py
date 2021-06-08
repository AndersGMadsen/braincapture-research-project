import numpy as np
from matplotlib import pyplot
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.patches as mpatches

# FONTS FOR PYPLOT
pyplot.rcParams['font.sans-serif'] = "Georgia"
pyplot.rcParams['font.family'] = "sans-serif"


## LOADING REAL DATA
X = np.load('/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/Data/X_artifacts_only.npy')
y = np.load('/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/Data/y_artifacts_only.npy')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

X_train = X_train.reshape(len(X_train), 19*25)
X_test = X_test.reshape(len(X_test), 19*25)


## LOADING FAKE IMAGES ##

fake_chew = np.load('/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/fake images/fake_chew_300.npy')
fake_chew = 10.0 * fake_chew - 5
fake_chew = fake_chew[:, :19, :25]

fake_elpp = np.load('/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/fake images/fake_elpp_200.npy')
fake_elpp = 10.0 * fake_elpp - 5
fake_elpp = fake_elpp[:, :19, :25]

fake_eyem = np.load('/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/fake images/fake_eyem_110.npy')
fake_eyem = 10.0 * fake_eyem - 5
fake_eyem = fake_eyem[:, :19, :25]

fake_musc = np.load('/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/fake images/fake_musc_170.npy')
fake_musc = 10.0 * fake_musc - 5
fake_musc = fake_musc[:, :19, :25]

fake_shiv = np.load('/home/williamtheodor/Documents/Fagpakke/epilepsy-project/GAN/fake images/fake_shiv_580.npy')
fake_shiv = 10.0 * fake_shiv - 5
fake_shiv = fake_shiv[:, :19, :25]


## ORGANIZING ARTIFACT DATA

fake_artifactsx = [fake_chew, fake_elpp, fake_eyem, fake_musc, fake_shiv]
fake_artifacts = []

for fake_artifact in fake_artifactsx:
    fake_artifacts.append(fake_artifact.reshape(len(fake_artifact), 19*25))

real_chew = X_test[y_test == 0]
real_elpp = X_test[y_test == 1]
real_eyem = X_test[y_test == 2]
real_musc = X_test[y_test == 3]
real_shiv = X_test[y_test == 4]

real_artifacts = [real_chew, real_elpp, real_eyem, real_musc, real_shiv]


## PLOTTING IMAGES

def plot(data, n=5):
    fig, ax = pyplot.subplots(n, n)
    for i in range(n * n):

        # turn off axis
        ax[i % n, i // n].axis('off')
        # plot raw pixel data
        im = ax[i % n, i // n].imshow(data[i, :, :, 0], cmap="magma")
        im.set_clim(-2, 2)

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)


plot(fake_eyem, 5)
pyplot.show()

#%%

## PCA
names = ['chew', 'elpp', 'eyem', 'musc', 'shiv']
artifact_names = ['Chewing', 'Electrode Pop', 'Eye Movement', 'Muscle', 'Shivering', 'Null']

#my_colors = ['yellow', 'red', 'lime', 'orange', 'purple']

#my_colors = ["#f7da65", "#ec656c", "#3fdc99", "#fd8f50", "#ab74b8", "#828bf2"]
my_colors = ["#f5cf40", "#e63f47", "#0ed280", "#fc7323", "#79218f", "#828bf2"]

chew_patch = mpatches.Patch(color=my_colors[0], label=names[0])
elpp_patch = mpatches.Patch(color=my_colors[1],label=names[1])
eyem_patch = mpatches.Patch(color=my_colors[2], label=names[2])
musc_patch = mpatches.Patch(color=my_colors[3], label=names[3])
shiv_patch = mpatches.Patch(color=my_colors[4], label=names[4])
X_patch = mpatches.Patch(color='blue', label='Real test data')

pca = PCA(n_components=5)
pca.fit(X_train)

pc_X_test = pca.transform(X_test)

pc_real_artifacts = []
pc_fake_artifacts = []

for real_artifact in real_artifacts:
    pc_real_artifacts.append(pca.transform(real_artifact))

for fake_artifact in fake_artifacts:
    pc_fake_artifacts.append(pca.transform(fake_artifact))

pc1 = 0
pc2 = 1

# PC PLOT FOR REAL DATA
fig = pyplot.figure()
ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])

ax.scatter(pc_X_test[:,pc1], pc_X_test[:,pc2], c="#828bf2")

for i, pc_real_artifact in enumerate(pc_real_artifacts):
    ax.scatter(pc_real_artifact[:,pc1], pc_real_artifact[:,pc2],color=my_colors[i])

ax.legend(handles=[chew_patch, elpp_patch, eyem_patch, musc_patch, shiv_patch, X_patch],
           bbox_to_anchor=(1.05, 1.0), loc='upper left')
pyplot.title("PCA Plot of Real Artifact Data", size=18)
pyplot.show()

# PC PLOT FOR FAKE DATA
fig = pyplot.figure()
ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])

ax.scatter(pc_X_test[:,pc1], pc_X_test[:,pc2], c="#828bf2")

for i, pc_fake_artifact in enumerate(pc_fake_artifacts):
    ax.scatter(pc_fake_artifact[:,pc1], pc_fake_artifact[:,pc2],color=my_colors[i])

ax.legend(handles=[chew_patch, elpp_patch, eyem_patch, musc_patch, shiv_patch, X_patch],
           bbox_to_anchor=(1.05, 1.0), loc='upper left')
pyplot.title("PCA Plot of Fake Artifact Data", size=18)
pyplot.show()

#%%
fig, ax = pyplot.subplots(2, 3, sharey=True)
fig.tight_layout()

for i in range(5):
    ax[i % 2, i % 3].scatter(pc_real_artifacts[i][:,pc1], pc_real_artifacts[i][:,pc2],color='blue')
    ax[i % 2, i % 3].scatter(pc_fake_artifacts[i][:,pc1], pc_fake_artifacts[i][:,pc2],color='red')
    ax[i % 2, i % 3].set_title(artifact_names[i])


pyplot.show()

#%%


fig, ax = pyplot.subplots(2, 3, sharey=True)

artifact_names = ['Chewing', 'Electrode Pop', 'Eye Movement', 'Muscle', 'Shivering', 'Null']

electrode_names = ['Fp1', 'F7', 'T3', 'T5', 'F3', 'C3', 'P3', 'O1', 'Cz', 'Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6', 'A1', 'A2']

for i, artifact in enumerate(fake_artifacts):
    x = (sum(artifact) / len(artifact)).reshape(19, 25)

    for j, row in enumerate(x):
        x[j] = np.convolve(row, np.ones(25)/25, mode='same')
    im = ax[i % 2, i % 3].imshow(x, cmap="magma")
    im.set_clim(-1, 1)
    ax[i % 2, i % 3].set_title(artifact_names[i])

    if i == 5:
        ax[i % 2, i % 3].set_visible(False)

fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
fig.colorbar(im, cax=cbar_ax)

pyplot.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.1)


fig.show()

#%%


