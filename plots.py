# Classics
import numpy as np
import pandas as pd
from tqdm import tqdm

# Plots
import seaborn as sns
import matplotlib.pyplot as plt



from sklearn.decomposition import PCA


# Sampling Methods
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import ClusterCentroids
from imblearn.combine import SMOTEENN 


# Fonts for pyplot
plt.rcParams['font.sans-serif'] = "Georgia"
plt.rcParams['font.family'] = "sans-serif"

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE) 



np.random.seed(26060000)
#%%

label_dict = {'chew': 0, 'elpp': 1, 'eyem': 2, 'musc': 3, 'shiv': 4, 'null': 5}

X = np.load("X1.npy")
y = np.load("Y1.npy")
groups = np.load("groups1.npy")

X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
nan_filter = ~np.isnan(X).any(axis=1)
X = X[nan_filter]
y = y[nan_filter]

shuffler = np.random.permutation(len(X))
X = X[shuffler]
y = y[shuffler]

part = 1/100
X = X[:int(part * len(X))]
y = y[:int(part * len(y))]



pca=PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)



X = pd.DataFrame(X, columns=['PC1', 'PC2'])

names = ['chew', 'elpp', 'eyem', 'musc', 'shiv', 'null']
#names = ['chewing', 'electrode pops', 'eye movements', 'muscle movements', 'shivering', 'null']
y = pd.DataFrame(y, columns=names).idxmax(1)

X_SMOTE, y_SMOTE = SMOTE().fit_resample(X, y)
X_SMOTEENN, y_SMOTEENN = SMOTEENN().fit_resample(X, y)
X_NearMiss, y_NearMiss = NearMiss().fit_resample(X, y)
X_ClusterCentroids, y_ClusterCentroids = ClusterCentroids().fit_resample(X, y)


data = pd.concat([X, y], axis=1).rename(columns={0: 'class'})
data_SMOTE = pd.concat([X_SMOTE, y_SMOTE], axis=1).rename(columns={0: 'class'})
data_SMOTEENN = pd.concat([X_SMOTEENN, y_SMOTEENN], axis=1).rename(columns={0: 'class'})
data_NearMiss= pd.concat([X_NearMiss, y_NearMiss], axis=1).rename(columns={0: 'class'})
data_ClusterCentroids= pd.concat([X_ClusterCentroids, y_ClusterCentroids], axis=1).rename(columns={0: 'class'})


#%%


datas = [data, data_SMOTE, data_SMOTEENN, data_NearMiss, data_ClusterCentroids]
titles = ['Original Data', 'Over Sampled w. SMOTE', 'Resampled w. SMOTEENN', 
          'Under Sampled w. NearMiss', 'Under Sampled w. ClusterCentroids']

for i, data in enumerate(datas):
    
    g = sns.scatterplot(
        data=data,
        x="PC1", y="PC2", hue="class",
    )
    g.set_title(titles[i])
    plt.xlim(-12, 19)
    plt.ylim(-12, 17)
    plt.show()


