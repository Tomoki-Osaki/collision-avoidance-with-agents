# https://www.sktime.net/en/v0.19.2/examples/02_classification.html

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sktime.datasets import load_basic_motions
from sklearn.metrics import accuracy_score
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.hybrid import HIVECOTEV2
from tqdm import tqdm
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

## KNeighborsTimeSeriesClassifier
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier as skKNTSC
classifier = skKNTSC(n_neighbors=1, weights='distance', distance="dtw")
classifier.fit(motions_train_X, motions_train_y)
y_pred = classifier.predict(motions_test_X)
accuracy_score(motions_test_y, y_pred)


from tslearn.neighbors import KNeighborsTimeSeriesClassifier as tsKNTSC
clf = tsKNTSC(n_neighbors=1, weights='distance')
clf.fit(motions_train_X, motions_train_y)
y_pred = clf.predict(motions_test_X)
accuracy_score(motions_test_y, y_pred)

arr1 = np.arange(0, 45., 1)
arr2 = np.arange(0, 50., 1)
arr3 = np.arange(0, 55., 1)
arr4 = np.arange(0, 58., 1)
arr5 = np.arange(0, 60., 1)
arr6 = np.arange(0, 63., 1)
arr7 = np.arange(0, 68., 1)
arr8 = np.arange(0, 75., 1)
arr9 = np.arange(0, 80., 1)
arr10 = np.arange(0, 100., 1)
arr11 = np.arange(0, 110., 1)

arrs = [arr1, arr2, arr3, arr4, arr5, arr6, arr7, arr8, arr9, arr10, arr11]
labs = np.array(
    ['short', 'short', 'short', 'short', 'short', 'short', 'med', 'med', 'med', 'long', 'long']
)

max_len = max(len(arr) for arr in arrs)

padded_arrs = np.array(
    [np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan) for arr in arrs]
)

print(padded_arrs.shape)
print(padded_arrs)

import random
from collections import Counter

res_ts = []
len_train = 7
for _ in tqdm(range(1000)):
    nums = [i for i in range(11)]
    random.shuffle(nums)
    
    train_x, train_y = padded_arrs[nums[:len_train]], labs[nums[:len_train]]
    test_x, test_y = padded_arrs[nums[len_train:]], labs[nums[len_train:]]
    
    clf_ts = tsKNTSC(n_neighbors=1, weights='uniform')
    clf_ts.fit(train_x, train_y)
    y_pred_ts = clf_ts.predict(test_x)
    res_ts.append(accuracy_score(test_y, y_pred_ts))

plt.hist(res_ts)
plt.title('tslean')
plt.show()

print('tslean', Counter(res_ts).most_common())

