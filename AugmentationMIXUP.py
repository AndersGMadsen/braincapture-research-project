import numpy as np
import torch
# Models
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# Score Metrics
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from tqdm import tqdm
#plot spectrogram
from matplotlib.pyplot import specgram

#mixup
import random

np.random.seed(3)

label_dict = {'chew': 0, 'elpp': 1, 'eyem': 2, 'musc': 3, 'shiv': 4, 'null': 5}
print('loading data')
X = np.load("/Users/yeganehghamari/epilepsy-project/data/X1.npy")
y = np.load("/Users/yeganehghamari/epilepsy-project/data/Y1.npy")
groups = np.load("/Users/yeganehghamari/epilepsy-project/data/groups1.npy")

X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
nan_filter = ~np.isnan(X).any(axis=1)
X = X[nan_filter]
y = y[nan_filter]

eyem = X[y[:,label_dict["eyem"]] == 1]
#null = X[(Y[:,label_dict["null"]] == 1) & (Y[:,label_dict["eyem"]] == 0)]
null = X[y[:,label_dict["eyem"]] != 1]
np.random.shuffle(null)
np.random.shuffle(eyem)

X_par = np.concatenate((null, eyem))
y_par = np.concatenate((np.zeros(len(null)),  np.ones(len(eyem))))


# we shuffle the data

shuffler = np.random.permutation(len(X_par))
X_par = X_par[shuffler]
y_par = y_par[shuffler]
print('data prepared')

# we split the data
X_split = int(0.8*len(X_par))
y_split = int(0.8*len(y_par))

X_train = X_par[:X_split]
y_train = y_par[:y_split]
X_test = X_par[X_split:]
y_test = y_par[y_split:]
print('data split done')


X_par = torch.tensor(X_par)
y_par = torch.tensor(y_par)
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)

#spectrogramMake(MNE_raw=X[0], t0=0, tWindow=120, crop_fq=45, FFToverlap=1, show_chan_num=19)
#plot_spec = plotSpec(ch_names=X_par['ch_names'], chan=show_chan_num,
                                #fAx=fAx[fAx <= crop_fq], tAx=tAx, Sxx=normSxx)
#plot_spec.show()
#MIXUP https://github.com/anhtuan85/Data-Augmentation-for-Object-Detection/blob/master/augmentation.ipynb
#image1 = Image.open("./data/000144.jpg", mode= "r")
#image1 = image1.convert("RGB")
#objects1= parse_annot("./data/000144.xml")
#boxes1 = torch.FloatTensor(objects1['boxes'])
#labels1 = torch.LongTensor(objects1['labels'])
#difficulties1 = torch.ByteTensor(objects1['difficulties'])
#draw_PIL_image(image1, boxes1, labels1)

#image2 = Image.open("./data/000055.jpg", mode= "r")
#image2 = image2.convert("RGB")
#objects2= parse_annot("./data/000055.xml")
#boxes2 = torch.FloatTensor(objects2['boxes'])
#labels2 = torch.LongTensor(objects2['labels'])
#difficulties2 = torch.ByteTensor(objects2['difficulties'])
#draw_PIL_image(image2, boxes2, labels2)


#image_info_1 = {"image": F.to_tensor(image1), "label": labels1, "box": boxes1, "difficult": difficulties1}
#image_info_2 = {"image": F.to_tensor(image2), "label": labels2, "box": boxes2, "difficult": difficulties2}


def mixup(eeg_info_1=None, eeg_info_2=None, lambd=None):

    X1 = eeg_info_1["eeg"]  # Tensor
    X2 = eeg_info_2["eeg"]  # Tensor

    y1 = eeg_info_1["label"]  # Tensor
    y2 = eeg_info_2["label"]

    mix_X = torch.zeros(X1.size())
    mix_X = X1 * lambd
    mix_X += X2 * (1. - lambd)

    mix_y = torch.zeros(y1.size())
    mix_p = y1 * lambd
    mix_p += y2 * (1. - lambd) #Sth bw 0-1
    if mix_p<=0.5:
        mix_y = 1
    else:
        mix_y = 0

    # mixup_width = max(img1.shape[2], img2.shape[2])
    # mix_up_height = max(img1.shape[1], img2.shape[1])

    # mix_img = torch.zeros(3, mix_up_height, mixup_width)
    # mix_img[:, :img1.shape[1], :img1.shape[2]] = img1 * lambd

    return mix_X, mix_y

mix_Xs = torch.zeros(5, len(X_train[0]))
mix_ys = torch.zeros(5)
#Mixup using 10 of training datasets

#choose mixup batch from the training data
indices = torch.randperm(len(X_train))[:10] #random generator without replacement
X_1 = X_train[indices[:5]]
X_2 = X_train[indices[5:]]

y_1 = y_train[indices[:5]]
y_2 = y_train[indices[5:]]

#choose from each group randomly
for i in range(len(y_1)):

    eeg_info_1 = {"eeg": X_1[i], "label": y_1[i]}
    eeg_info_2 = {"eeg": X_2[i], "label": y_2[i]}
    lambd = random.uniform(0, 1)
    mix_X, mix_y = mixup(eeg_info_1=eeg_info_1, eeg_info_2=eeg_info_2, lambd=lambd)
    mix_Xs[i,:] = mix_X
    mix_ys[i] = mix_y
    print("Lambda: ", lambd)

print('mix_label:', mix_ys)

#add mixups to traindataset
X_train_res = torch.cat([mix_Xs, X_train], dim = 0)
y_train_res = torch.cat([mix_ys, y_train])

X_train_res = X_train_res.numpy()
y_train_res = y_train_res.numpy()
part = 1 / 100
X_train_res = X_train_res[:int(part * len(X_train_res))]
y_train_res = y_train_res[:int(part * len(y_train_res))]

#specgram(X_1[0], NFFT=25, Fs=19, noverlap=0)
#specgram(X_2[0], NFFT=25, Fs=19, noverlap=0)
#specgram(mix_Xs[0], NFFT=25, Fs=19, noverlap=0)

scaler = StandardScaler().fit(X_train_res)
X_train_res = scaler.transform(X_train_res)
X_test = scaler.transform(X_test)

print('models start')
models = [RandomForestClassifier(n_jobs=3), MLPClassifier(), LinearDiscriminantAnalysis()]


for model in models:
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    print('model: ', model)
    print("Train data: Accuracy:", np.mean(model.predict(X_train_res) == y_train_res) * 100)

    print("Test data: Accuracy:", np.mean(model.predict(X_test) == y_test) * 100)
    print("Test data: Balanced accuracy:", balanced_accuracy_score(y_test, y_pred) * 100)
    print("Test data: F1-Score:", f1_score(y_test, y_pred) * 100)
    print("Test: Weighted F1-Score:", round(f1_score(y_test, y_pred, average='weighted') * 100, 3), '%')
    print()

