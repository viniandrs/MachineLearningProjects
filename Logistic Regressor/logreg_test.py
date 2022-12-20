import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics 

import medmnist
from medmnist import INFO
import torchvision.transforms as transforms

# constants and global variabels-----------------------------------
NUMFEATURES = 784
NUMEPOCHS = 1000
BATCH_SIZE = 128
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
x_train_unorm = np.reshape(training_images, (n_rows_train, NUMFEATURES))
x_test_unorm = np.reshape(test_dataset.imgs, (n_rows_test, NUMFEATURES))

# normalizing
scaler = StandardScaler()
scaler.fit(x_train_unorm)

x_train = (x_train_unorm - scaler.mean_) / np.sqrt(scaler.var_ + 0.000000001)
y_train = np.reshape(training_labels, (n_rows_train))

# normalizing
scaler.fit(x_test_unorm)
x_test = (x_test_unorm - scaler.mean_) / np.sqrt(scaler.var_ + 0.000000001)
y_test = np.reshape(test_dataset.labels, (n_rows_test))
        
# -----------------------------------------------------------------

# training the model ----------------------------------------------

model = LogisticRegression(penalty='l2',            
                            C=0.1,                  
                            class_weight='balanced',
                            max_iter=NUMEPOCHS)     

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
#cm_display.plot()

# Compute ROC curve and ROC area for each class
y_score = trained_model.decision_function(x_test)
fpr = list()
tpr = list()
roc_auc = list()
fpr, tpr, _ = metrics.roc_curve(y_test, y_score)
roc_auc = metrics.auc(fpr, tpr)
    
plt.figure()
lw = 2
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.2f)" % roc_auc,
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")

# printing the results
print(f"accuracy: {accuracy}")
print(f"f-1 score: {f1}")
print(f"balanced accuracy: {balanced_acc}")

plt.show()
