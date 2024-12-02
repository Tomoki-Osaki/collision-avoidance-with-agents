import numpy as np
import matplotlib.pyplot as plt
from tslearn.datasets import CachedDatasets
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
    
seed = 0
np.random.seed(seed)
X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
X_train = X_train[y_train < 4]  # Keep first 3 classes
np.random.shuffle(X_train)
    
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train[:50])
X_train = TimeSeriesResampler(sz=40).fit_transform(X_train)
sz = X_train.shape[1]

print('Euclidean k-means')
km = TimeSeriesKMeans(n_clusters=3, verbose=True, random_state=seed)
y_pred = km.fit_predict(X_train)

fig, ax = plt.subplots()
for yi in range(3):
    plt.subplot(3, 3, yi + 1)
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("Euclidean $k$-means")
plt.show()

n = 3
km = TimeSeriesKMeans(n_clusters=3, verbose=True, random_state=seed)
labels_euclidean = km.fit_predict(X_train)

# 標準化したデータのプロット
fig, axes = plt.subplots(n, figsize=(8.0, 12.0))
for i in range(n):
    ax = axes[i]
    # データのプロット
    for xx in X_train[labels_euclidean == i]:
        ax.plot(xx.ravel(), "k-", alpha=.2)
    # 重心のプロット
    ax.plot(km.cluster_centers_[i].ravel(), "r-")
    # 軸の設定とテキストの表示
    ax.set_xlim(0, 24)
    datanum = np.count_nonzero(labels_euclidean == i)
plt.show()


# find the elbow point
def find_elbow(array):
    distortions = []
    for n in range(1, 11):
        km = TimeSeriesKMeans(n_clusters=n, verbose=True, random_state=seed)
        km.fit_predict(array)
        distortions.append(km.inertia_)
        
    plt.plot(range(1, 11), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()

