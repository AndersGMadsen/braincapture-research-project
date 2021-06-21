#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 11:56:47 2021

@author: williamtheodor
"""
import numpy as np
from keras.datasets.mnist import load_data
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

from torchvision import datasets, transforms, utils



def load_images():
# load real images
    (realX, realy), (testX, testy) = load_data()
    realX = realX.reshape(-1, 28*28)
    testX = testX.reshape(-1, 28*28)

    # load fake images
    fakeX = np.array([])
    fakey = np.array([])

    for i in range(10):
        for image in np.load("fake_%d.npy" % i).reshape(100, 28*28):
            fakeX = np.append(fakeX, image)
            fakey = np.append(fakey, i)
    
    fakeX = fakeX.reshape(1000, 28*28)
    
    shuffler = np.random.permutation(len(realX))
    realX = realX[shuffler]
    realy = realy[shuffler]
    
    # shuffle data
    shuffler = np.random.permutation(len(fakeX))
    fakeX = fakeX[shuffler]
    fakey = fakey[shuffler]
    

    
    return realX, realy, fakeX, fakey, testX, testy
    
def run(n_real, n_fake):
    
    realX, realy, fakeX, fakey, testX, testy = load_images()
    
    trainX = np.concatenate((realX[:n_real], fakeX[:n_fake]))
    trainy = np.concatenate((realy[:n_real], fakey[:n_fake]))
    
    shuffler = np.random.permutation(len(trainX))
    trainX = trainX[shuffler]
    trainy = trainy[shuffler]

    RF = RandomForestClassifier()
    RF.fit(trainX, trainy)
    
    return RF.score(testX, testy)
    
from tqdm import tqdm


n_real, n_fake = 200, 0

scores = []

for i in tqdm(range(10)):
    scores.append(run(n_real, n_fake))

print()
print(sum(scores) / len(scores))
    
    
    
    