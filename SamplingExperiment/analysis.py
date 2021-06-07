import numpy as np
import random
from sklearn.utils import resample
from sklearn.metrics import classification_report

y = np.load(r"C:\Users\andersgm\Documents\Courses\02466 Project work F21\Project\multiclass-y.npy")
ypred = np.load(r"C:\Users\andersgm\Documents\Courses\02466 Project work F21\Project\epilepsy-project\SamplingExperiment\Results\predictions_LDA_1_1_5_5_123213_21-04-21_13-00-48.npy")
seed = 123213
np.random.seed(seed)
random.seed(seed)

mask = resample(np.arange(len(y)), replace=False, n_samples=int(len(y) * 0.1), stratify=y)
y = y[mask]

label_dict = {'chew': 0, 'elpp': 1, 'eyem': 2, 'musc': 3, 'shiv': 4, 'null': 5}
print(classification_report(y, ypred.flatten(), target_names=list(label_dict.keys())))
print()