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
    
    for n in range(1, 2):
        for i in range(n+1, len(a)-n*2-1):
            if np.all(a[i-n-1:i] == a[i-n-1]) and np.all(a[i-n-1] == a[i+n:i+2*n+1]):
                a[i:i+n] = a[i-n-1]
            
    for n in range(4, 7):
        mask = np.zeros(len(a))
        for i in range(len(a)-n):
            if np.all(a[i:i+n] == a[i]):
                mask[i:i+n] = True
                
        for i in range(len(a)):
            if not mask[i]:
                a[i] = 5
                
    for n in range(4, 7):
        mask = np.zeros(len(a))
        for i in range(len(a)-n):
            if np.all(a[i:i+n] == a[i]):
                mask[i:i+n] = True
                
        for i in range(len(a)):
            if not mask[i]:
                a[i] = 5   
                
    return a

def clean2(a):
    a = deepcopy(a)
    for _ in range(10):
        for n in range(2, 5):
            for i in range(n+1, len(a)-n*2-1):
                if np.all(a[i-n-1:i] == a[i-n-1]) and np.all(a[i-n-1] == a[i+n:i+2*n+1]):
                    a[i:i+n] = a[i-n-1]
        
    for _ in range(10):
        for n in range(4, 20):
            mask = np.zeros(len(a))
            for i in range(len(a)-n):
                if np.all(a[i:i+n] == a[i]):
                    mask[i:i+n] = True
                    
            for i in range(len(a)):
                if not mask[i]:
                    a[i] = 5
    
    for _ in range(10):
        for n in range(2, 9):
            for i in range(n+1, len(a)-n*2-1):
                if np.all(a[i-n-1:i] == a[i-n-1]) and np.all(a[i-n-1] == a[i+n:i+2*n+1]):
                    a[i:i+n] = a[i-n-1]
            
    return a



def plot_artifacts(ax, y, dataframe):
    for i in range(len(dataframe)-1):    
        width = dataframe["Start"][i+1] - dataframe["Start"][i]
        x_pos = dataframe["Start"][i]
        color = color_dict[y[i]]
        ax.add_patch(Rectangle((x_pos, 0), width, 1, facecolor=color, fill=True))
    
    ax.tick_params(left =False)
    

#%%
color_dict = {0: "#f5cf40", 1: "#e63f47", 2: "#0ed280", 3: "#fc7323", 4: "#79218f", 5: "#828bf2"} 

patient_name = np.random.choice( np.unique(df["Name"]))
#patient_name = "00000254_s005_t000"
#patient_name = "00008181_s003_t001"
#patient_name = "00009362_s001_t001"
#patient_name = "00007823_s001_t001"
#patient_name = "00004473_s002_t001"
patient_name = "00010212_s001_t000"
patient = deepcopy(df[df["Name"] == patient_name])
patient = patient.sort_values(by="Start", ascending=True, key=lambda col: col.values).reset_index(drop=True)
prediction = ypred[patient["index"].values]

fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)

ax[0].plot([0, 0],[0, 1], color='white')
plot_artifacts(ax[0], patient["Multiclass label"], patient)
ax[0].set_ylabel("True")

ax[1].plot([0, 0],[0, 1], color='white')
plot_artifacts(ax[1], prediction, patient)
ax[1].set_ylabel("Predicted")

ax[2].plot([0, 0],[0, 1], color='white')
plot_artifacts(ax[2], clean(prediction), patient)
ax[2].set_ylabel("Simple Filter")

# ax[3].plot([0, 0],[0, 1], color='white')
# plot_artifacts(ax[3], clean2(prediction), patient)
# ax[3].set_ylabel("Advanced Filter")



plt.tight_layout()
plt.show()
     #%%