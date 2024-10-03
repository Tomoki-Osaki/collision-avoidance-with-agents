import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import myfuncs as mf
from gc import collect as g
from typing import Literal

SUBJECTS = mf.SUBJECTS
CONDITIONS = mf.CONDITIONS # [urgent, nonurgent, omoiyari]
AGENTS = mf.AGENTS
NUM_AGENTS = mf.NUM_AGENTS
TRIALS = mf.TRIALS

df_all = mf.make_dict_of_all_info()

mf.plot_traj_per_trials(df_all, 3, 'nonurgent', 10)
mf.plot_traj_compare_conds(df_all, 1, 5)

df_part = mf.make_df_trial(df_all, 3, 'nonurgent', 10, 2)
cols = df_part.columns

mf.plot_distance(df_all, 3, 20, "dist_actual_ideal")

df_clustering = mf.make_df_for_clustering(5, 20, "dist_actual_ideal")
true_labels = ["omoiyari", "isogi", "yukkuri"] * 10

from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesScalerMeanVariance    
from tslearn.metrics import dtw 

time_np = to_time_series_dataset(df_clustering.T)

n = 3
km_euclidean = TimeSeriesKMeans(n_clusters=n, metric='euclidean', random_state=42)
labels_euclidean = km_euclidean.fit_predict(df_clustering.T)
print(labels_euclidean)

fig = plt.figure(figsize=(12, 6))
for i in range(n):
    ax = fig.add_subplot(1, 3, i+1)
    for x in time_np[labels_euclidean == i]:
        ax.plot(x.ravel(), 'k-', alpha=0.2)
    ax.plot(km_euclidean.cluster_centers_[i].ravel(), 'r-')

    datanum = np.count_nonzero(labels_euclidean == i)
    ax.text(0.5, (0.7+0.25), f'Cluster{(i)} : n = {datanum}')
plt.suptitle('time series clustering')
plt.show()


distortions = [] 
for i in range(1, 11): 
    ts_km = TimeSeriesKMeans(n_clusters=i, metric="dtw", random_state=42) 
    ts_km.fit_predict(df_clustering.T) 
    distortions.append(ts_km.inertia_) 

plt.plot(range(1, 11), distortions, marker="o") 
plt.xticks(range(1, 11)) 
plt.xlabel("Number of clusters") 
plt.ylabel("Distortion") 
plt.show()


