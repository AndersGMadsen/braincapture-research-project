import os
import numpy as np
import mne
import matplotlib.pyplot as plt

# SETTINGS FOR PYPLOT
plt.rc('text', usetex=True)

font = {'family': 'serif',

        'size': '20',

        'serif': ['Computer Modern'],

        'sans-serif': ['Computer Modern']}

plt.rc('font', **font)

plt.rc('axes', titlesize=28, labelsize=26)

plt.rc('xtick', labelsize=20)

plt.rc('ytick', labelsize=20)

plt.rc('legend', fontsize=16)

'''sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_filt-0-40_raw.fif')'''

sample_data_raw_file = '/home/williamtheodor/Documents/Fagpakke/artifact_dataset/01_tcp_ar/002/00000254/s005_2010_11_15/00000254_s005_t000.edf'
raw = mne.io.read_raw_edf(sample_data_raw_file)


#raw.plot_psd(fmax=50)

fig = raw.plot(duration=5, start=200, n_channels=10, color='b')
fig.suptitle(r"\textbf{Raw EEG Waves}", size=48, y=1.04)
fig.savefig("Raw EEG waves", dpi=1000, bbox_inches = 'tight')
