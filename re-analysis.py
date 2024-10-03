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
    
df_clustering = pd.concat([omoiyari, urgent, nonurgent])

from tslearn.clustering import TimeSeriesKMeans
import collections

#metric : ユークリッド距離(euclidean) , DTW(dtw)
metric = 'euclidean'
# cluster数
n_clusters = 3
#クラスタリングの実装
tskm_base = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric,
                             max_iter=100, random_state=42)
tskm_base.fit(df_clustering.T.values)

#クラスタリングごとの表示
cnt = collections.Counter(tskm_base.labels_)
clusters = list(tskm_base.labels_)
cluster_labels = {}
for k in cnt:
    cluster_labels['cluster-{}'.format(k)] = cnt[k]
# クラスターごとの数
print(sorted(cluster_labels.items()))
