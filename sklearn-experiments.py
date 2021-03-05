import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
np.random.seed(26060000)
#%%

label_dict = {'chew': 0, 'elpp': 1, 'eyem': 2, 'musc': 3, 'shiv': 4, 'null': 5}

X = np.load("X1.npy")
Y = np.load("Y1.npy")
groups = np.load("groups1.npy")

X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
nan_filter = ~np.isnan(X).any(axis=1)
X = X[nan_filter]
Y = Y[nan_filter]

eyem = X[Y[:,label_dict["eyem"]] == 1]
#null = X[(Y[:,label_dict["null"]] == 1) & (Y[:,label_dict["eyem"]] == 0)]
null = X[Y[:,label_dict["chew"]] != 1]
np.random.shuffle(null)
np.random.shuffle(eyem)
null = null[:len(eyem)]

Xpar = np.concatenate((null, eyem))
Ypar = np.repeat((0,1), len(eyem))

#%%

shuffler = np.random.permutation(len(Xpar))

Xpar = Xpar[shuffler]
Ypar = Ypar[shuffler]

split = int(0.8*len(Xpar))

Xtrain = Xpar[:split]
Ytrain = Ypar[:split]
Xtest = Xpar[split:]
Ytest = Ypar[split:]

#%%

scaler = StandardScaler().fit(Xtrain)
Xtrain = scaler.transform(Xtrain)
Xtest = scaler.transform(Xtest)
   
#%%

clf = RandomForestClassifier(n_jobs=3)
clf.fit(Xtrain, Ytrain)

#%%
print("Train accuracy:", np.mean(clf.predict(Xtrain) == Ytrain)*100)
print("Test accuracy:", np.mean(clf.predict(Xtest) == Ytest)*100)
















