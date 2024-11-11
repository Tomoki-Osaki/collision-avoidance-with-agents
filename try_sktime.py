# https://www.sktime.net/en/v0.19.2/examples/02_classification.html

# Plotting and data loading imports used in this notebook
import matplotlib.pyplot as plt
from sktime.datasets import (
    load_arrow_head,
    load_basic_motions,
    load_japanese_vowels,
    load_plaid,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sktime.classification.kernel_based import RocketClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sktime.classification.interval_based import DrCIF
from sktime.transformations.panel.compose import ColumnConcatenator
from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.classification.feature_based import RandomIntervalClassifier
from sktime.transformations.panel.padder import PaddingTransformer
from sktime.classification.hybrid import HIVECOTEV2


# Load arrow head dataset, pandas multiindex format, also accepted by sktime classifiers
arrow_train_X, arrow_train_y = load_arrow_head(
    split="train", return_type="pd-multiindex"
)
arrow_test_X, arrow_test_y = load_arrow_head(split="test", return_type="pd-multiindex")
print(arrow_train_X.head())

# Load arrow head dataset in nested pandas format, also accepted by sktime classifiers
arrow_train_X, arrow_train_y = load_arrow_head(split="train", return_type="nested_univ")
arrow_test_X, arrow_test_y = load_arrow_head(split="test", return_type="nested_univ")
arrow_train_X.iloc[:5]


#################################################################
# "basic motions" dataset
motions_X, motions_Y = load_basic_motions(return_type="numpy3d")
motions_train_X, motions_train_y = load_basic_motions(
    split="train", return_type="numpy3d"
)
motions_test_X, motions_test_y = load_basic_motions(split="test", return_type="numpy3d")
print(type(motions_train_X))
print(
    motions_train_X.shape,
    motions_train_y.shape,
    motions_test_X.shape,
    motions_test_y.shape,
)
plt.title(" First and second dimensions of the first instance in BasicMotions data")
plt.plot(motions_train_X[0][0])
plt.plot(motions_train_X[0][1])
plt.show()

# loads both train and test together
vowel_X, vowel_y = load_japanese_vowels()
print(type(vowel_X))

plt.title(" First two dimensions of two instances of Japanese vowels")
plt.plot(vowel_X.iloc[0, 0], color="b")
plt.plot(vowel_X.iloc[1, 0], color="b")
plt.plot(vowel_X.iloc[0, 1], color="r")
plt.plot(vowel_X.iloc[1, 1], color="r")
plt.show()

plaid_X, plaid_y = load_plaid()
plaid_train_X, plaid_train_y = load_plaid(split="train")
plaid_test_X, plaid_test_y = load_plaid(split="test")
print(type(plaid_X))

plt.title(" Four instances of PLAID dataset")
plt.plot(plaid_X.iloc[0, 0])
plt.plot(plaid_X.iloc[1, 0])
plt.plot(plaid_X.iloc[2, 0])
plt.plot(plaid_X.iloc[3, 0])
plt.show()

### ROCKET multivariate time series analysis
rocket = RocketClassifier(num_kernels=2000)
rocket.fit(arrow_train_X, arrow_train_y)
y_pred = rocket.predict(arrow_test_X)

accuracy_score(arrow_test_y, y_pred)

### HIVECOTEV2
hc2 = HIVECOTEV2(time_limit_in_minutes=0.2)
hc2.fit(arrow_train_X, arrow_train_y)
y_pred = hc2.predict(arrow_test_X)

accuracy_score(arrow_test_y, y_pred)


cross_val_score(rocket, arrow_train_X, y=arrow_train_y, cv=KFold(n_splits=4))

knn = KNeighborsTimeSeriesClassifier()
param_grid = {"n_neighbors": [1, 5], "distance": ["euclidean", "dtw"]}
parameter_tuning_method = GridSearchCV(knn, param_grid, cv=KFold(n_splits=4))

parameter_tuning_method.fit(arrow_train_X, arrow_train_y)
y_pred = parameter_tuning_method.predict(arrow_test_X)

accuracy_score(arrow_test_y, y_pred)

calibrated_drcif = CalibratedClassifierCV(
    base_estimator=DrCIF(n_estimators=10, n_intervals=5), cv=4
)

calibrated_drcif.fit(arrow_train_X, arrow_train_y)
y_pred = calibrated_drcif.predict(arrow_test_X)

accuracy_score(arrow_test_y, y_pred)

from sklearn.model_selection import KFold, cross_val_score

cross_val_score(rocket, arrow_train_X, y=arrow_train_y, cv=KFold(n_splits=4))



### Multivariate time series classification ###
# many classifiers, including ROCKET, and HC2, are configured to work with multivariate input as follows.
rocket = RocketClassifier(num_kernels=2000)
rocket.fit(motions_train_X, motions_train_y)
y_pred = rocket.predict(motions_test_X)

accuracy_score(motions_test_y, y_pred)


HIVECOTEV2(time_limit_in_minutes=0.2)
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

padded_clf = PaddingTransformer() * RandomIntervalClassifier(n_intervals=5)
padded_clf.fit(plaid_train_X, plaid_test_y)
y_pred = padded_clf.predict(plaid_test_X)

accuracy_score(plaid_test_y, y_pred)

