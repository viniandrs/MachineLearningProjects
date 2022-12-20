import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier as knn
import sklearn.metrics as metrics 

import medmnist
from medmnist import INFO
import torchvision.transforms as transforms

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
    transforms.ToTensor(),
])

# raw data
train_dataset = DataClass(
    split='train', transform=data_transform, download=True)
val_dataset = DataClass(
    split='val', transform=data_transform, download=True)
test_dataset = DataClass(
        split='test', transform=data_transform, download=True)

# -----------------------------------------------------------------

# processing data -------------------------------------------------

# concatenating the val and train dataset as a single dataset
training_images = np.concatenate(
    (train_dataset.imgs, val_dataset.imgs))
training_labels = np.concatenate(
    (train_dataset.labels, val_dataset.labels))

# converting images to arrays
n_rows_train = training_images.shape[0]
n_rows_test = test_dataset.imgs.shape[0]

# normalizers
x_train_unorm = np.reshape(training_images, (n_rows_train, NUMFEATURES))
scaler_train = StandardScaler()
scaler_train.fit(x_train_unorm)

x_test_unorm = np.reshape(test_dataset.imgs, (n_rows_test, NUMFEATURES))
scaler_test = StandardScaler()
scaler_test.fit(x_test_unorm)

x_train = (x_train_unorm - scaler_train.mean_) / np.sqrt(scaler_train.var_ + 0.000000001)
y_train = np.reshape(training_labels, (n_rows_train))

x_test = (x_test_unorm - scaler_test.mean_) / np.sqrt(scaler_test.var_ + 0.000000001)
y_test = np.reshape(test_dataset.labels, (n_rows_test))
        
# -----------------------------------------------------------------

# training the model ----------------------------------------------

model = knn(n_neighbors=1,            
            p=2,                  
            n_jobs=-1)     

trained_model = model.fit(x_train, y_train)
y_predict = trained_model.predict(x_test)

# -----------------------------------------------------------------

# metrics
accuracy = metrics.accuracy_score(y_test, y_predict)
f1 = metrics.f1_score(y_test, y_predict)
balanced_acc = metrics.balanced_accuracy_score(y_test, y_predict)

# confusion matrix
cm_display = metrics.ConfusionMatrixDisplay.from_predictions(y_test, 
                                                             y_predict,
                                                             display_labels = ['cancer', 'normal'])

# printing the results
print(f"accuracy: {accuracy}")
print(f"f-1 score: {f1}")
print(f"balanced accuracy: {balanced_acc}")

plt.show()
