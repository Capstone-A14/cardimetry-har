from sklearn import svm, metrics
from sklearn.model_selection import train_test_split

import pickle
import yaml



# Labels and features
cmhar_labels    = ['L-Si', 'Si-L', 'Si-St', 'St-Si', 'Walk', 'Fall']
cmhar_feats     = [
    'rms_qw', 'rms_qx', 'rms_qy', 'rms_qz',
    'median_qw', 'median_qx', 'median_qy', 'median_qz', 
    'cov_ww', 'cov_wx', 'cov_wy', 'cov_wz', 'cov_xx', 'cov_xy', 'cov_xz', 'cov_yy', 'cov_yz', 'cov_zz',
    'relent_w', 'relent_x', 'relent_y', 'relent_z'
]



# Read all the features .yaml
cmhar_features = {}

for label in cmhar_labels:
    cmhar_features[label] = []

    file_path = f'data/ExtFeat/{label}_feats.yaml'
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    for i in range(len(data['rms_qw'])):
        point = []

        for feat in cmhar_feats:
            point.append(data[feat][i])

        cmhar_features[label].append(point)



# Configure the datasets for one-to-rest SVM model
cmhar_data = {}

for label_base in cmhar_labels:
    cmhar_data[label_base] = {'data': [], 'target': []}

    for label_other in cmhar_labels:
        data_len = len(cmhar_features[label_other])
        cmhar_data[label_base]['data'] += cmhar_features[label_other]

        if label_base == label_other:
            cmhar_data[label_base]['target'] += [1 for __ in range(data_len)]

        else:
            cmhar_data[label_base]['target'] += [0 for __ in range(data_len)]



# Split dataset into training and test set
X_train = {}
X_test  = {}
y_train = {}
y_test  = {}

for label in cmhar_labels:
    X_train[label]  = None
    X_test[label]   = None
    y_train[label]  = None
    y_test[label]   = None

    X_train[label], X_test[label], y_train[label], y_test[label] = train_test_split(
        cmhar_data[label]['data'], 
        cmhar_data[label]['target'], 
        test_size       = 0.3,
        train_size      = 0.7,
        random_state    = 69
    )



# Create classifier
cmhar_clf = {
    'L-Si'  : pickle.load(open('model/L-Si_model.sav', 'rb')),
    'Si-L'  : pickle.load(open('model/Si-L_model.sav', 'rb')),
    'Si-St' : pickle.load(open('model/Si-St_model.sav', 'rb')),
    'St-Si' : pickle.load(open('model/St-Si_model.sav', 'rb')),
    'Walk'  : pickle.load(open('model/Walk_model.sav', 'rb')),
    'Fall'  : pickle.load(open('model/Fall_model.sav', 'rb'))
}



# Calculate accuracy
for label in cmhar_labels:
    y_pred = cmhar_clf[label].predict(X_test[label])
    print(f'{label} accuracy: {metrics.accuracy_score(y_test[label], y_pred)}')