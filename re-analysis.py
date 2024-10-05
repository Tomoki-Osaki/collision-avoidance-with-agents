import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from gc import collect as g
from typing import Literal
import math
import myfuncs as mf

SUBJECTS = mf.SUBJECTS
CONDITIONS = mf.CONDITIONS # [urgent, nonurgent, omoiyari]
AGENTS = mf.AGENTS
NUM_AGENTS = mf.NUM_AGENTS
TRIALS = mf.TRIALS

df_all = mf.make_dict_of_all_info(SUBJECTS)

ID = 11
agents = 20
cond = "nonurgent"
trial = 4

# %% plot functions
mf.plot_traj_per_trials(df_all, ID, cond, agents)
mf.plot_traj_compare_conds(df_all, ID, agents)

mf.plot_dist_compare_conds(df_all, ID, agents, "dist_actual_ideal")
mf.plot_dist_per_cond(df_all, ID, agents, "dist_actual_ideal")

mf.plot_dist_compare_conds(df_all, ID, agents, "closest_dists")
mf.plot_dist_per_cond(df_all, ID, agents, "closest_dists")

# %%
df_trial = mf.make_df_trial(df_all, ID, cond, agents, trial)
cols = df_trial.columns

posXtplus1  = pd.concat([df_trial.posX[1:], pd.Series([None])])
posYtplus1  = pd.concat([df_trial.posY[1:], pd.Series([None])])
df_trial['posXt+1'] = posXtplus1
df_trial['posYt+1'] = posYtplus1

def _calc_distance(myX, myY, anotherX, anotherY):
    mypos = np.array([myX, myY])
    anotherpos = np.array([anotherX, anotherY])
    distance = np.linalg.norm(mypos - anotherpos)
    
    return distance

def calc_deg(x0, y0, x1, y1, x2, y2):
    #角度計算開始
    try:
        vec1 = [x1 - x0, y1 - y0]
        vec2 = [x2 - x0, y2 - y0]
    except TypeError:
        return None
    
    absvec1 = np.linalg.norm(vec1)
    absvec2 = np.linalg.norm(vec2)
    inner = np.inner(vec1, vec2)
    cos_theta = inner / (absvec1 * absvec2)
    theta = math.degrees(math.acos(cos_theta))
    
    return theta

# %% time series clustering
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

