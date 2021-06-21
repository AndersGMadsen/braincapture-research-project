# BrainCapture Research Project
Here is the repository for the BrainCapture Research Project at the Technical University of Denmark.
The repository consists of three folders. In the Experiments folder is the description and code needed to reproduce the experiments along with the results. In the Data Analysis folder is the code used to analyse the data and generate visualization. This code is unorganized and uncommented. Lastly, in the PilotScript folder, the pilot scripts can be seen that was used to research before the main experiments.

## Data
In order to run the experiments, you need to download the TUH EEG Artifact Corpus, which is freely available at Temple University here: https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml

Afterwards, you need to run the preprocessing of the raw EEG files. In this project, we used the preprocessing made by David NyrnBerg which can be found here: https://github.com/DavidEnslevNyrnberg/DTU_DL_EEG

## Experiment: Sampling
In the sampling experiment folder, there is a single file experiment.py, that is needed to run the experiment. The experiment performs a repeat two-level cross-validation with hyperparameter optimization. The code is made to run each fold in parallel. Here is an example of how the experiment is run with the LDA learning algorithm and random undersampling with SMOTE.

```python
python experiment.py --x multiclass_X_new.npy --y multiclass_y_new.npy --groups multiclass_patients_new.npy --model LDA --technique 4 --n_repeats 5 --seed 55784899 --optimize 25 --n_parallel 25 --logging LDA_4_16-06-21_12-14-22.out
```
The input parameters are as follows:
- **x** : Preprocessed EEG data
- **y** : Labels for the preprocessed EEG data
- **groups** : Patients groups in order to make StratifedGroupKFold Cross-Validation
- **model** : The learning algorithm used in the experiment. Hyperparameter space is already defined.
- **technique** : Sampling method
- **n_repeat** : Number of times to repeat the cross-validation
- **seed** : The seed used for the experiment
- **Optimize** : Number of models trained in the hyperparameter optimization
- **n_parallel** : Number of cores to use
- **logging** : Logfile

There are more parameters which can be seen in the code.


## Experiment: Mixup

## Experiment: GAN
In the GAN folder, a the few scripts that are necissary to genrated fake data can be found. First, the file TUH_GAN_v4_server.py trains the GAN on data from one of the artifacts and saves the model parameters in a folder generator_models. The artifact as well as the number of epochs to train the models must be specified. For example,

```python
python TUH_GAN_v4_server.py --artifact 2 --epochs 100
```
trains a GAN on Eye Movement data for 100 epochs.

Specifically, the input parameters are:
- **artifact**: An aritfact (integer) in range 0-4: {0: chew, 1: elpp, 2: eyem, 3: musc, 4: shiv}.
- **epochs**: An integer desribing the number of epochs to train the GAN (at least 100 is recommended).

When a generator is trained, the file generate_images.py can be run to generate new and augmented data. Inside the script, the exact path to the saved model parameters as well as a path describing where to save the new data need to be specified. Now, the GAN generated can be explored!
