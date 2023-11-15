import os
import numpy as np
import matplotlib.pyplot as plt



# Variables
TIMESTAMP   = 0
QW_DATA     = 1
QX_DATA     = 2
QY_DATA     = 3
QZ_DATA     = 4

cmhar_labels    = ['L-Si', 'Si-L', 'Si-St', 'St-Si', 'Walk', 'Fall', 'Actual']
cmhar_datasets  = {}



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



# Seek for data
label       = 'L-Si'
idx         = 3
plotdata    = cmhar_datasets[label][idx]

plt.plot(plotdata[TIMESTAMP], plotdata[QW_DATA], label='$q_w$')
plt.plot(plotdata[TIMESTAMP], plotdata[QX_DATA], label='$q_x$')
plt.plot(plotdata[TIMESTAMP], plotdata[QY_DATA], label='$q_y$')
plt.plot(plotdata[TIMESTAMP], plotdata[QZ_DATA], label='$q_z$')

plt.title(f'Data seek of {label} No.{idx}')
plt.xlabel('Time (ms)')
plt.ylabel('Quaternion Unit')
plt.legend()
plt.grid()
plt.show()