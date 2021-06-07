import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
from copy import deepcopy
import os
from scipy import stats
os.chdir(r"C:\Users\andersgm\Documents\Courses\02466 Project work F21\Project")
label_dict = {'chew': 0, 'elpp': 1, 'eyem': 2, 'musc': 3, 'shiv': 4, 'null': 5}
color_dict = {0: "yellow", 1: "red", 2: "lime", 3: "orange", 4: "purple", 5: "cornflowerblue"}

#%%

df = pd.read_pickle("dataframe_float32.pkl")
df["index"] = df.index

ypreds = np.load("predictions_LDA_4_5_5_5_55784899_04-06-21_20-08-15.npy")
#ypred = np.load("predictions_LDA_2_1_5_5_123213_03-06-21_21-14-26.npy")
#ypred = np.load("predictions_LDA_2_5_5_5_55784899_04-06-21_03-42-55.npy")
ypred = stats.mode(ypreds, axis=0).mode[0]


#%%
def clean(a):
    a = deepcopy(a)
    for i in range(2, len(a)-2):
        if a[i-2] == a[i-1] and a[i-1] == a[i+1] and a[i+1] == a[i+2]:
            a[i] = a[i-1]
        
    for i in range(3, len(a)-4):
        if a[i-3] == a[i-2] and a[i-2] == a[i-1] and a[i-1] == a[i+2] and a[i+2] == a[i+3] and a[i+3] == a[i+4]:
            a[i] = a[i-1]
            a[i+1] = a[i-1]
            
    mask = np.zeros(len(a))
    for i in range(len(a)-4):
        if np.all(a[i:i+4] == a[i]):
            mask[i:i+4] = 1
            
    for i in range(len(a)):
        if not mask[i]:
            a[i] = 5
    
    # mask = np.zeros(len(a))
    # for i in range(len(a)-5):
    #     if np.all(a[i:i+5] == a[i]):
    #         mask[i:i+5] = 1
            
    # for i in range(len(a)):
    #     if not mask[i]:
    #         a[i] = 5
    
    # a = deepcopy(a)
    # mask = np.zeros(len(a)) 
    # for i in range(len(a)-6):
    #     if np.all(a[i:i+6] == a[i]):
    #         mask[i:i+6] = 1
            
    # for i in range(len(a)):
    #     if not mask[i]:
    #         a[i] = 5
    
    return a

def clean2(a):
    a = deepcopy(a)
    for i in range(2, len(a)-2):
        if a[i-2] == a[i-1] and a[i-1] == a[i+1] and a[i+1] == a[i+2]:
            a[i] = a[i-1]
        
    for i in range(3, len(a)-4):
        if a[i-3] == a[i-2] and a[i-2] == a[i-1] and a[i-1] == a[i+2] and a[i+2] == a[i+3] and a[i+3] == a[i+4]:
            a[i] = a[i-1]
            a[i+1] = a[i-1]
            
    mask = np.zeros(len(a))
    for i in range(len(a)-4):
        if np.all(a[i:i+4] == a[i]):
            mask[i:i+4] = 1
            
    for i in range(len(a)):
        if not mask[i]:
            a[i] = 5
     
    for n in range(1, 8):
        for i in range(n, len(a)-n*2):
            if np.all(a[i-n:i] == 5) and np.all(a[i+n:i+n*2] == 5) and np.all(a[i:i+n] == 1):
                a[i:i+2] = 5
    
    return a


def plot_artifacts(ax, y, dataframe):
    for i in range(len(dataframe)-1):    
        width = dataframe["Start"][i+1] - dataframe["Start"][i]
        x_pos = dataframe["Start"][i]
        color = color_dict[y[i]]
        ax.add_patch(Rectangle((x_pos, 0), width, 1, facecolor=color, fill=True))
    
    ax.tick_params(left =False)
    
    

#%%

patient_name = np.random.choice( np.unique(df["Name"]))
patient_name = "00000254_s005_t000"
#patient_name = "00009362_s001_t001"
patient = deepcopy(df[df["Name"] == patient_name])
patient = patient.sort_values(by="Start", ascending=True, key=lambda col: col.values).reset_index(drop=True)
prediction = ypred[patient["index"].values]

fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True)

ax[0].plot([0, 0],[0, 1], color='white')
plot_artifacts(ax[0], patient["Multiclass label"], patient)
ax[0].set_ylabel("True")

ax[1].plot([0, 0],[0, 1], color='white')
plot_artifacts(ax[1], prediction, patient)
ax[1].set_ylabel("Predicted")

ax[2].plot([0, 0],[0, 1], color='white')
plot_artifacts(ax[2], clean(prediction), patient)
ax[2].set_ylabel("Simple Filter")

ax[3].plot([0, 0],[0, 1], color='white')
plot_artifacts(ax[3], clean2(clean(prediction)), patient)
ax[3].set_ylabel("Advanced Filter")



plt.tight_layout()
plt.show()
#%%