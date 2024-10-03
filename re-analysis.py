import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import myfuncs as mf
from gc import collect as g

SUBJECTS = mf.SUBJECTS
CONDITIONS = mf.CONDITIONS # [urgent, nonurgent, omoiyari]
AGENTS = mf.AGENTS
NUM_AGENTS = mf.NUM_AGENTS
TRIALS = mf.TRIALS

df_all = mf.make_dict_of_all_info()

mf.plot_traj_per_trials(df_all, 1, 'urgent', 5)
mf.plot_traj_compare_conds(df_all, 1, 5)

df_part = mf.make_df_trial(df_all, 3, 'nonurgent', 10, 2)
cols = df_part.columns

def plot(ID, agents, dist):
    fig = plt.figure(figsize=(10, 6), tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    for cond, color in zip(CONDITIONS, mcolors.TABLEAU_COLORS):
        df_small = df_all[f'ID{ID}'][cond][f'agents{agents}']
        for tri in TRIALS:
            if tri == TRIALS[0]:
                ax.plot(df_small[f'trial{tri}'][dist], color=color, alpha=.7, label=cond)
            else:
                ax.plot(df_small[f'trial{tri}'][dist], color=color, alpha=.7)
    ax.set_title(f"{dist} ID{ID} agents{agents}")
    plt.legend()
    plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)

df_clustering = pd.DataFrame()
for trial in TRIALS:
    omoiyari = df_all["ID1"]["omoiyari"]["agents20"][f"trial{trial}"]["dist_actual_ideal"]
    urgent = df_all["ID1"]["urgent"]["agents20"][f"trial{trial}"]["dist_actual_ideal"]
    nonurgent = df_all["ID1"]["nonurgent"]["agents20"][f"trial{trial}"]["dist_actual_ideal"]
    
    df_clustering[f"omoiyari_dist_{trial}"] = omoiyari
    df_clustering[f"urgent_dist_{trial}"] = urgent
    df_clustering[f"nonurgent_dist_{trial}"] = nonurgent
    
    df_clustering.dropna(inplace=True)

from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesScalerMeanVariance    
from tslearn.metrics import dtw 
time_np = to_time_series_dataset(df_clustering)

n = 3
km_euclidean = TimeSeriesKMeans(n_clusters=n, metric='euclidean', random_state=0)
labels_euclidean = km_euclidean.fit_predict(df_clustering)
print(labels_euclidean)

fig, axes = plt.subplots(n, figsize=(8,16))
for i in range(n):
    ax = axes[i]

    for x in time_np[labels_euclidean == i]:
        ax.plot(x.ravel(), 'k-', alpha=0.2)
    ax.plot(km_euclidean.cluster_centers_[i].ravel(), 'r-')

    datanum = np.count_nonzero(labels_euclidean == i)
    ax.text(0.5, (0.7+0.25), f'Cluster{(i)} : n = {datanum}')
    if i == 0:
        ax.set_title('time series clustering')
plt.show()

distortions = [] 
for i in range(1,11): 
    ts_km = TimeSeriesKMeans(n_clusters=i,metric="dtw",random_state=42) 
    ts_km.fit_predict(df_clustering) 
    distortions.append(ts_km.inertia_) 

plt.plot(range(1,11),distortions,marker="o") 
plt.xticks(range(1,11)) 
plt.xlabel("Number of clusters") 
plt.ylabel("Distortion") 
plt.show()

ts_km = TimeSeriesKMeans(n_clusters=3, metric="dtw", random_state=42) 
y_pred = ts_km.fit_predict(df_clustering)

plt.figure()
for yi in range(3):
    plt.subplot(3, 3, yi + 1)
    # for xx in X_train[y_pred == yi]:
    #     plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(ts_km.cluster_centers_[yi].ravel(), "r-")
    # plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
