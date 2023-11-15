from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from data.cmhar_funcs import *

import numpy as np
import matplotlib.pyplot as plt
import pickle
import yaml



# Create classifier
cmhar_clf = {
    'L-Si'  : pickle.load(open('model/L-Si_model.sav', 'rb')),
    'Si-L'  : pickle.load(open('model/Si-L_model.sav', 'rb')),
    'Si-St' : pickle.load(open('model/Si-St_model.sav', 'rb')),
    'St-Si' : pickle.load(open('model/St-Si_model.sav', 'rb')),
    'Walk'  : pickle.load(open('model/Walk_model.sav', 'rb')),
    'Fall'  : pickle.load(open('model/Fall_model.sav', 'rb'))
}



# Read actual data taken
actual_data = np.loadtxt(
    fname       = 'data/Actual/har_dataset0.csv',
    delimiter   = ','
).T



# Feature extraction
rms = cmharCalcRms(actual_data)
med = cmharCalcMedian(actual_data)
cov = cmharCalcCovariance(actual_data)



# Convert to points
points = []
for i in range(len(rms)):
    point   = []
    point   += list(rms[i]) + list(med[i]) + list(cov[i])
    points.append(point)



# Pass to model
iterate     = []
lsi_preds   = []
sil_preds   = []
sist_preds  = []
stsi_preds  = []
walk_preds  = []
fall_preds  = []
for i in range(len(points)):
    iterate.append(i)
    lsi_preds.append(cmhar_clf['L-Si'].predict([points[i]]))
    sil_preds.append(cmhar_clf['Si-L'].predict([points[i]]))
    sist_preds.append(cmhar_clf['Si-St'].predict([points[i]]))
    stsi_preds.append(cmhar_clf['St-Si'].predict([points[i]]))
    walk_preds.append(cmhar_clf['Walk'].predict([points[i]]))
    fall_preds.append(cmhar_clf['Fall'].predict([points[i]]))


plt.plot(iterate, lsi_preds, label='L-Si')
plt.plot(iterate, sil_preds, label='Si-L')
plt.plot(iterate, sist_preds, label='Si-St')
plt.plot(iterate, stsi_preds, label='St-Si')
plt.plot(iterate, walk_preds, label='Walk')
plt.plot(iterate, fall_preds, label='Fall')
plt.legend()
plt.show()