from sklearn import svm, metrics
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
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
    'L-Si'  : svm.SVC(
        kernel  ='rbf',
        degree  = 4,
        gamma   = 16
    ),
    'Si-L'  : svm.SVC(
        kernel  ='rbf',
        degree  = 4,
        gamma   = 7.5
    ),
    'Si-St' : svm.SVC(
        kernel  ='rbf',
        degree  = 4,
        gamma   = 19
    ),
    'St-Si' : svm.SVC(
        kernel  ='rbf',
        degree  = 4,
        gamma   = 14
    ),
    'Walk'  : svm.SVC(
        kernel  ='rbf',
        degree  = 4,
        gamma   = 9
    ),
    'Fall'  : svm.SVC(
        kernel  ='rbf',
        degree  = 4,
        gamma   = 9
    )
}

for label in cmhar_labels:
    cmhar_clf[label].fit(X_train[label], y_train[label])

    file_name = f'model/{label}_model.sav'
    pickle.dump(cmhar_clf[label], open(file_name, 'wb'))




# Calculate accuracy
# cm = []
for label in cmhar_labels:
    y_pred = cmhar_clf[label].predict(X_test[label])
    # cm.append(metrics.confusion_matrix(y_test[label], y_pred))
    print(f'{label} accuracy: {metrics.accuracy_score(y_test[label], y_pred)}')

# disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm[1], display_labels=[0, 1])
# disp.plot()
# plt.show()