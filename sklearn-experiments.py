# Classics
import numpy as np

# Models
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# Score Metrics
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

# Sampling Methods
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import ClusterCentroids

from tqdm import tqdm
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

eyem = X[y[:,label_dict["eyem"]] == 1]
#null = X[(Y[:,label_dict["null"]] == 1) & (Y[:,label_dict["eyem"]] == 0)]
null = X[y[:,label_dict["eyem"]] != 1]
np.random.shuffle(null)
np.random.shuffle(eyem)



#null = null[:len(eyem)]    # down sampling?

X_par = np.concatenate((null, eyem))
y_par = np.concatenate((np.zeros(len(null)),  np.ones(len(eyem))))


# we shuffle the data

shuffler = np.random.permutation(len(X_par))
X_par = X_par[shuffler]
y_par = y_par[shuffler]

# we split the data
X_split = int(0.8*len(X_par))
y_split = int(0.8*len(y_par))

X_train = X_par[:X_split]
y_train = y_par[:y_split]
X_test = X_par[X_split:]
y_test = y_par[y_split:]


'''we under sample only the train data using MissNear'''
'''
X_train_res, y_train_res = NearMiss().fit_resample(X_train, y_train)

shuffler = np.random.permutation(len(X_train_res))
X_train_res = X_train_res[shuffler]
y_train_res = y_train_res[shuffler]

# for now, we only use a small part of data
part = 1 / 20
X_train_res = X_train_res[:int(part * len(X_train_res))]
y_train_res = y_train_res[:int(part * len(y_train_res))]

# %%

scaler = StandardScaler().fit(X_train_res)
X_train_res = scaler.transform(X_train_res)
X_test = scaler.transform(X_test)
'''
# we up sample only the train data using SMOTE

X_train_res, y_train_res = SMOTE().fit_resample(X_train, y_train)

shuffler = np.random.permutation(len(X_train_res))
X_train_res = X_train_res[shuffler]
y_train_res = y_train_res[shuffler]

# for now, we only use a small part of data
part = 1 / 20
X_train_res = X_train_res[:int(part * len(X_train_res))]
y_train_res = y_train_res[:int(part * len(y_train_res))]

# %%

scaler = StandardScaler().fit(X_train_res)
X_train_res = scaler.transform(X_train_res)
X_test = scaler.transform(X_test)


'''undersample cluster centroids'''

'''
X_train_res, y_train_res = ClusterCentroids().fit_resample(X_train, y_train)

shuffler = np.random.permutation(len(X_train_res))
X_train_res = X_train_res[shuffler]
y_train_res = y_train_res[shuffler]

# for now, we only use a small part of data
part = 1 / 20
X_train_res = X_train_res[:int(part * len(X_train_res))]
y_train_res = y_train_res[:int(part * len(y_train_res))]

# %%

scaler = StandardScaler().fit(X_train_res)
X_train_res = scaler.transform(X_train_res)
X_test = scaler.transform(X_test)
'''

models = [RandomForestClassifier(n_jobs=3), MLPClassifier(), LinearDiscriminantAnalysis()]

for model in models:
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    print('model: ', model)
    print("Train data: Accuracy:", np.mean(model.predict(X_train_res) == y_train_res) * 100)

    print("Test data: Accuracy:", np.mean(model.predict(X_test) == y_test) * 100)
    print("Test data: Balanced accuracy:", balanced_accuracy_score(y_test, y_pred) * 100)
    print("Test data: F1-Score:", f1_score(y_test, y_pred) * 100)
    #print("Test data: F1-Score:", roc_auc_score(y_test, model.decision_function(X_test)) * 100)
    print()















