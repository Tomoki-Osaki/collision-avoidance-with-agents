# https://www.sktime.net/en/v0.19.2/examples/02_classification.html

import numpy as np
import matplotlib.pyplot as plt
from sktime.datasets import load_basic_motions
from sklearn.metrics import accuracy_score
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.hybrid import HIVECOTEV2
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
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
classifier = KNeighborsTimeSeriesClassifier(distance="dtw")
classifier.fit(motions_train_X, motions_train_y)
y_pred = classifier.predict(motions_test_X)

accuracy_score(motions_test_y, y_pred)


# サンプルデータを生成
data_A = np.arange(100)  # 長さ100のデータA
data_B = np.arange(80)   # 長さ80のデータB（基準）
data_C = np.arange(120)  # 長さ120のデータC

# ダウンサンプリングの関数
def downsample(data, target_length):
    indices = np.linspace(0, len(data) - 1, target_length, dtype=int)
    return data[indices]

# データAとデータCを長さ80にダウンサンプリング
data_A_downsampled = downsample(data_A, len(data_B))
data_C_downsampled = downsample(data_C, len(data_B))

# 結果を表示
print("Data A (downsampled):", data_A_downsampled)
print("Data B:", data_B)
print("Data C (downsampled):", data_C_downsampled)
