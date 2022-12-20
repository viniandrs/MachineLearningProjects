from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

import medmnist
from medmnist import INFO
import torchvision.transforms as transforms

import numpy as np
import pandas as pd

# constants and global variabels-----------------------------------
NUMFEATURES = 784
# -----------------------------------------------------------------

# importing breastmnist -------------------------------------------

info = INFO['breastmnist']
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])

# preprocessing
data_transform = transforms.Compose([
    transforms.ToTensor()
])

# raw data
train_dataset = DataClass(
    split='train', transform=data_transform, download=True)
val_dataset = DataClass(
    split='val', transform=data_transform, download=True)

# -----------------------------------------------------------------
 
 # preprocessing data -------------------------------------------------

# concatenating the val and train dataset as a single dataset
training_images = np.concatenate(
    (train_dataset.imgs, val_dataset.imgs))
training_labels = np.concatenate(
    (train_dataset.labels, val_dataset.labels))

# converting images to arrays
n_rows = training_images.shape[0]
x_train_unorm = np.reshape(training_images, (n_rows, NUMFEATURES))

# normalizing
scaler = StandardScaler()
scaler.fit(x_train_unorm)

x_train = (x_train_unorm - scaler.mean_) / np.sqrt(scaler.var_ + 0.000000001)
y_train = np.reshape(training_labels, (n_rows))

# -----------------------------------------------------------------
 
 # cross-validating ------------------------------------------------

parameters = {
    'n_neighbors':[x for x in range(1, 6)], # regularization paramater
    'p': [x for x in range(1, 6)], # minkowski order
    'n_jobs': [-1]
}

classifier = GridSearchCV(estimator=knn(), 
                          param_grid=parameters, 
                          scoring='balanced_accuracy', 
                          cv=5, 
                          n_jobs=-1, 
                          return_train_score=True)
classifier.fit(x_train, y_train)

# -----------------------------------------------------------------

# printing the results
results = classifier.cv_results_
df = pd.DataFrame.from_dict(results)
df.to_csv(r'knn_cv_results.csv', index=False)