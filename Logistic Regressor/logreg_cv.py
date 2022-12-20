import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import medmnist
from medmnist import INFO
import torchvision.transforms as transforms

# constants and global variabels-----------------------------------
NUMFEATURES = 784
NUMEPOCHS = 1000
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
x_unorm = np.reshape(training_images, (n_rows, NUMFEATURES))

# normalizing
scaler = StandardScaler()
scaler.fit(x_unorm)

x_train = (x_unorm - scaler.mean_) / np.sqrt(scaler.var_ + 0.000000001)
y_train = np.reshape(training_labels, (n_rows))
        
# -----------------------------------------------------------------

# cross-validating ------------------------------------------------

parameters = {
    'C':[np.power(10.0, x - 5.0) for x in range(21)], # regularization paramater
    'class_weight': ['balanced', None],
    'max_iter': [NUMEPOCHS]
}

classifier = GridSearchCV(estimator=LogisticRegression(), 
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
df.to_csv(r'my_data.csv', index=False)
