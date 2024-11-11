# https://www.sktime.net/en/v0.19.2/examples/02_classification.html

# Plotting and data loading imports used in this notebook
import matplotlib.pyplot as plt
from sktime.datasets import load_basic_motions
from sklearn.metrics import accuracy_score
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.interval_based import DrCIF
from sktime.transformations.panel.compose import ColumnConcatenator
from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.classification.hybrid import HIVECOTEV2

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
rocket = RocketClassifier(num_kernels=2000)
rocket.fit(motions_train_X, motions_train_y)
y_pred = rocket.predict(motions_test_X)

accuracy_score(motions_test_y, y_pred)


hc2 = HIVECOTEV2(time_limit_in_minutes=0.2)
hc2.fit(motions_train_X, motions_train_y)
y_pred = hc2.predict(motions_test_X)

accuracy_score(motions_test_y, y_pred)


"""
1.Concatenation of time series columns into a single long time series column via 
ColumnConcatenator and apply a classifier to the concatenated data,

2.Dimension ensembling via ColumnEnsembleClassifier in which one classifier is 
fitted for each time series column/dimension of the time series and their 
predictions are combined through a voting scheme.
"""

clf = ColumnConcatenator() * DrCIF(n_estimators=10, n_intervals=5)
clf.fit(motions_train_X, motions_train_y)
y_pred = clf.predict(motions_test_X)

accuracy_score(motions_test_y, y_pred)

col = ColumnEnsembleClassifier(
    estimators=[
        ("DrCIF0", DrCIF(n_estimators=10, n_intervals=5), [0]),
        ("ROCKET3", RocketClassifier(num_kernels=1000), [3]),
    ]
)

col.fit(motions_train_X, motions_train_y)
y_pred = col.predict(motions_test_X)

accuracy_score(motions_test_y, y_pred)

