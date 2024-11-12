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
print('motions_test_X.shape', motions_test_X.shape)
print('motions_test_y.shape', motions_test_y.shape)

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


from sktime.distances import dtw_distance
x_1d = np.array([1, 2, 3, 4])  # 1d array
y_1d = np.array([5, 6, 7, 8, 9, 10])  # 1d array
dtw_distance(x_1d, y_1d)


# 元データの長さとダウンサンプリング後の長さ
original_length = 120
downsampled_length = 80

# 元のタイムポイント（1/120, 2/120, ..., 120/120）
original_timepoints = np.linspace(1, original_length, original_length) / original_length

# ダウンサンプリング後のタイムポイント（1/80, 2/80, ..., 80/80）
downsampled_timepoints = np.linspace(1, downsampled_length, downsampled_length) / downsampled_length

# タイムポイントの線形補間
interpolated_timepoints = np.interp(np.linspace(0, len(original_timepoints) - 1, downsampled_length),
                                    np.arange(len(original_timepoints)), original_timepoints)

print("Original Timepoints:", original_timepoints)
print("Downsampled Timepoints:", downsampled_timepoints)
print("Interpolated Timepoints:", interpolated_timepoints)

# 元データのタイムポイント（0から1の範囲に正規化）
relative_timepoints = np.linspace(0, 1, original_length)

# ダウンサンプリング後のタイムポイント（0から1の範囲に正規化）
downsampled_relative_timepoints = np.linspace(0, 1, downsampled_length)

# 相対的タイムポイントの線形補間
interpolated_relative_timepoints = np.interp(np.linspace(0, len(relative_timepoints) - 1, downsampled_length),
                                             np.arange(len(relative_timepoints)), relative_timepoints)

print("Original Relative Timepoints:", relative_timepoints)
print("Downsampled Relative Timepoints:", downsampled_relative_timepoints)
print("Interpolated Relative Timepoints:", interpolated_relative_timepoints)

a = [i**2/100 for i in range(50)]
b = [i**2/100 for i in range(100)]

c = [i**2/100 for i in range(0, 100, 2)]
d = [i**2/100 for i in range(70)]


a = [i**2/100 for i in range(30)]
b = [i**2/100 for i in range(90)]

c = [i**2/100 for i in range(0, 90, 3)]

a = [i for i in range(30)]
b = [i for i in range(90)]

c = [i for i in range(0, 90, 3)]
