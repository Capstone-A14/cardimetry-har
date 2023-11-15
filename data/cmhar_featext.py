import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
from cmhar_funcs import *



# Variables
TIMESTAMP   = 0
QW_DATA     = 1
QX_DATA     = 2
QY_DATA     = 3
QZ_DATA     = 4

cmhar_labels    = ['L-Si', 'Si-L', 'Si-St', 'St-Si', 'Walk', 'Fall']
cmhar_features  = [
    'rms_qw', 'rms_qx', 'rms_qy', 'rms_qz',
    'median_qw', 'median_qx', 'median_qy', 'median_qz', 
    'cov_ww', 'cov_wx', 'cov_wy', 'cov_wz', 'cov_xx', 'cov_xy', 'cov_xz', 'cov_yy', 'cov_yz', 'cov_zz',
    'relent_w', 'relent_x', 'relent_y', 'relent_z'
]
cmhar_featit    = ['rms', 'median', 'cov']
cmhar_datasets  = {}
cmhar_extracts  = {}



# Load all dataset and create extracted features datasets
for label in cmhar_labels:
    cmhar_datasets[label] = []
    datasets_num = len(os.listdir(label))
    
    for idx in range(datasets_num):
        cmhar_datasets[label].append(np.loadtxt(
                fname       = f'{label}/har_dataset{idx}.csv',
                delimiter   = ','
            ).T
        )

        # Normalize time
        offset = cmhar_datasets[label][idx][TIMESTAMP][0]
        cmhar_datasets[label][idx][TIMESTAMP] = cmhar_datasets[label][idx][TIMESTAMP] - offset



# Take feature for every window with length 3s (approx. 150 data points) and minimum overlap of 50%
for label in cmhar_labels:
    cmhar_extracts[label] = {}
    datasets_num = len(os.listdir(label))

    for feature in cmhar_features:
        cmhar_extracts[label][feature] = []

    for idx in range(datasets_num):
    
        for feature in cmhar_featit:

            if feature == 'rms':
                res = cmharCalcRms(cmhar_datasets[label][idx])
                for val in res:
                    cmhar_extracts[label]['rms_qw'].append(val[0].item()) 
                    cmhar_extracts[label]['rms_qx'].append(val[1].item()) 
                    cmhar_extracts[label]['rms_qy'].append(val[2].item()) 
                    cmhar_extracts[label]['rms_qz'].append(val[3].item()) 

            elif feature == 'median':
                res = cmharCalcMedian(cmhar_datasets[label][idx])
                for val in res:
                    cmhar_extracts[label]['median_qw'].append(val[0].item())
                    cmhar_extracts[label]['median_qx'].append(val[1].item())
                    cmhar_extracts[label]['median_qy'].append(val[2].item())
                    cmhar_extracts[label]['median_qz'].append(val[3].item())

            elif feature == 'cov':
                res = cmharCalcCovariance(cmhar_datasets[label][idx])
                for val in res:
                    cmhar_extracts[label]['cov_ww'].append(val[0].item())
                    cmhar_extracts[label]['cov_wx'].append(val[1].item())
                    cmhar_extracts[label]['cov_wy'].append(val[2].item())
                    cmhar_extracts[label]['cov_wz'].append(val[3].item())
                    cmhar_extracts[label]['cov_xx'].append(val[4].item())
                    cmhar_extracts[label]['cov_xy'].append(val[5].item())
                    cmhar_extracts[label]['cov_xz'].append(val[6].item())
                    cmhar_extracts[label]['cov_yy'].append(val[7].item())
                    cmhar_extracts[label]['cov_yz'].append(val[8].item())
                    cmhar_extracts[label]['cov_zz'].append(val[9].item())
                    cmhar_extracts[label]['relent_w'].append(val[10].item())
                    cmhar_extracts[label]['relent_x'].append(val[11].item())
                    cmhar_extracts[label]['relent_y'].append(val[12].item())
                    cmhar_extracts[label]['relent_z'].append(val[13].item())



# Write the extracted features on ExtFeat/
for label in cmhar_labels:

    file_path = f'ExtFeat/{label}_feats.yaml'

    with open(file_path, 'w') as file:
        yaml.dump(cmhar_extracts[label], file)