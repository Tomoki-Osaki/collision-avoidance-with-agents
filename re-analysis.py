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

df_all = mf.make_dict_of_all_info(SUBJECTS)

ID = 11
agents = 20
cond = "nonurgent"
trial = 2

mf.plot_traj_per_trials(df_all, ID, cond, agents)
mf.plot_traj_compare_conds(df_all, ID, agents)

df_trial = mf.make_df_trial(df_all, ID, cond, agents, trial)
cols = df_trial.columns

mf.plot_dist_compare_conds(df_all, ID, agents, "dist_actual_ideal")
mf.plot_dist_per_cond(df_all, ID, agents, "dist_actual_ideal")

mf.plot_dist_compare_conds(df_all, ID, agents, "closest_dists")
mf.plot_dist_per_cond(df_all, ID, agents, "closest_dists")



df_clustering = mf.make_df_for_clustering(df_all, 8, 20, "dist_actual_ideal")
true_labels = ["omoiyari", "isogi", "yukkuri"] * 8

from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesScalerMeanVariance    
from tslearn.metrics import dtw 

from sklearn.preprocessing import StandardScaler
scaler_std = StandardScaler()
df_clustering = scaler_std.fit_transform(df_clustering)


time_np = to_time_series_dataset(df_clustering.T)

n = 3
km_euclidean = TimeSeriesKMeans(n_clusters=n, metric='euclidean', random_state=42)
labels_euclidean = km_euclidean.fit_predict(df_clustering.T)
print(labels_euclidean)

fig = plt.figure(figsize=(12, 4))
for i in range(n):
    ax = fig.add_subplot(1, 3, i+1)
    for x in time_np[labels_euclidean == i]:
        ax.plot(x.ravel(), 'k-', alpha=0.2)
    ax.plot(km_euclidean.cluster_centers_[i].ravel(), 'r-')

    datanum = np.count_nonzero(labels_euclidean == i)
    ax.text(0.5, (0.7+0.25), f'Cluster{(i)} : n = {datanum}')
plt.suptitle('time series clustering')
plt.show()




def find_proper_num_clusters(df):
    distortions = [] 
    for i in range(1, 11): 
        ts_km = TimeSeriesKMeans(n_clusters=i, metric="dtw", random_state=42) 
        ts_km.fit_predict(df.T) 
        distortions.append(ts_km.inertia_) 
    
    plt.plot(range(1, 11), distortions, marker="o") 
    plt.xticks(range(1, 11)) 
    plt.xlabel("Number of clusters") 
    plt.ylabel("Distortion") 
    plt.show()
    
find_proper_num_clusters(df_clustering)



# df_a = pd.DataFrame()
# df_b = pd.DataFrame()
# for ID in SUBJECTS:
#     for cond in CONDITIONS:
#         df_tmp = pd.read_csv(f"04_RawData/{ID}_{cond}.csv")
#         df_tmp["condition"] = cond
#         df_tmp["ID"] = ID
#         df_b = pd.concat([df_b, df_tmp], ignore_index=True)
#     df_a = pd.concat([df_a, df_b], ignore_index=True)
# df_a.dropna(inplace=True, axis=1)
# df_a = df_a.query("trial != 0")
# df_a["agents"] = None



