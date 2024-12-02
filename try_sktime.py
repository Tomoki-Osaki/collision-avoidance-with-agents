# https://www.sktime.net/en/v0.19.2/examples/02_classification.html

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sktime.datasets import load_basic_motions
from sklearn.metrics import accuracy_score, classification_report
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.hybrid import HIVECOTEV2
from tqdm import tqdm
import random
from collections import Counter
import warnings 
warnings.simplefilter('ignore')

# "basic motions" dataset
motions_X, motions_Y = load_basic_motions(return_type="numpy3d")
motions_train_X, motions_train_y = load_basic_motions(split="train", return_type="numpy3d")
motions_test_X, motions_test_y = load_basic_motions(split="test", return_type="numpy3d")

print('motions_train_X.shape', motions_train_X.shape) # 40 people, 6 variables, 100 timepoints
print('motions_train_y.shape', motions_train_y.shape)
print('motions_test_X.shape ', motions_test_X.shape)
print('motions_test_y.shape ', motions_test_y.shape)

plt.title("First and second dimensions of the first instance in BasicMotions data")
plt.plot(motions_train_X[0][0])
plt.plot(motions_train_X[0][1])
plt.show()

### Multivariate time series classification motions data ###
# many classifiers, including ROCKET, and HC2, are configured to work with 
# multivariate input as follows.

## ROCKET
rocket = RocketClassifier(num_kernels=2000)
rocket.fit(motions_train_X, motions_train_y)
y_pred = rocket.predict(motions_test_X)

accuracy_score(motions_test_y, y_pred)

## HIVECOTEV2
hc2 = HIVECOTEV2(time_limit_in_minutes=0.2)
hc2.fit(motions_train_X, motions_train_y)
y_pred = hc2.predict(motions_test_X)

accuracy_score(motions_test_y, y_pred)

## KNeighborsTimeSeriesClassifier from sktime (CANNOT handle missing values)
# shape (40, 6, 100) = (data, variables, timepoints)
# each variable has 100 timepoints
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier as skKNTSC
classifier = skKNTSC(n_neighbors=1, weights='distance', distance="dtw")
classifier.fit(motions_train_X, motions_train_y)
y_pred = classifier.predict(motions_test_X)
report = pd.DataFrame(
    classification_report(motions_test_y, y_pred, digits=3, output_dict=True)
)
print(classification_report(motions_test_y, y_pred, digits=3))

## KNeighborsTimeSeriesClassifier from tslean (CAN handle missing values)
# shape (40, 100, 6) = (data, timepoints, variables)
# each timepoint has 6 variables
from tslearn.neighbors import KNeighborsTimeSeriesClassifier as tsKNTSC
clf = tsKNTSC(n_neighbors=1, weights='distance', metric='dtw')

train_X, test_X = np.zeros((40, 100, 6)), np.zeros((40, 100, 6))
for i in range(40):
    tmp1 = motions_train_X[i].T
    tmp2 = motions_test_X[i].T
    train_X[i] = tmp1
    test_X[i] = tmp2

clf.fit(train_X, motions_train_y)
y_pred = clf.predict(test_X)
report = pd.DataFrame(
    classification_report(motions_test_y, y_pred, digits=3, output_dict=True)
)
print(classification_report(motions_test_y, y_pred, digits=3))

