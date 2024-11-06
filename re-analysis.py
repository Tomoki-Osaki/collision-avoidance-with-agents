# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
#plt.rcParams['font.family'] = "MS Gothic"
import seaborn as sns

from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset

from sklearn import metrics
from sklearn.preprocessing import StandardScaler

import warnings
# warnings.simplefilter('ignore')
from gc import collect as g
from collections import Counter
from tqdm import tqdm
import time
import myfuncs as mf

SUBJECTS = mf.SUBJECTS
CONDITIONS = mf.CONDITIONS # [urgent, nonurgent, omoiyari]
AGENTS = mf.AGENTS
NUM_AGENTS = mf.NUM_AGENTS
TRIALS = mf.TRIALS

#%%
df_all = mf.make_dict_of_all_info(SUBJECTS)

# %% plot functions
ID = 23
agents = 20
mf.plot_traj_per_trials(df_all, ID, "omoiyari", agents)
mf.plot_traj_compare_conds(df_all, ID, agents)

mf.plot_dist_compare_conds(df_all, ID, agents, "dist_actual_ideal")
mf.plot_dist_per_cond(df_all, ID, agents, "dist_actual_ideal")

mf.plot_dist_compare_conds(df_all, ID, agents, "dist_closest")
mf.plot_dist_per_cond(df_all, ID, agents, "dist_closest")

mf.plot_dist_compare_conds(df_all, ID, agents, "dist_top12_closest")
mf.plot_dist_per_cond(df_all, ID, agents, "dist_top12_closest")

mf.plot_dist_compare_conds(df_all, ID, agents, "dist_from_start")
mf.plot_dist_per_cond(df_all, ID, agents, "dist_from_start")

mf.plot_all_dist_compare_conds(df_all, SUBJECTS, agents, "dist_actual_ideal")
mf.plot_all_dist_compare_conds(df_all, SUBJECTS, agents, "dist_from_start")

# %% time series clustering
dist = "dist_actual_ideal"
dist = "dist_from_start"
dist = "dist_closest"
dist = "dist_top12_closest"
n_clusters = 3

true_labels = ["omoiyari", "isogi", "yukkuri"] * 8

# scaler_std = StandardScaler()
# df_clustering = scaler_std.fit_transform(df_clustering)

# because clusters for each subject are not related, this way of counting is not proper and must need to consider anothey way!!!
df_labels = pd.DataFrame(columns=["true_labels", "clustered_labels"])
for ID in tqdm(SUBJECTS):
    df_clustering = mf.make_df_for_clustering(df_all, ID, 20, dist)
    km_euclidean = TimeSeriesKMeans(n_clusters=n_clusters, metric='dtw', random_state=2)
    labels_euclidean = km_euclidean.fit_predict(df_clustering.T)
        
    df_tmp = pd.DataFrame({"true_labels": true_labels,
                           "clustered_labels": labels_euclidean})
    
    df_labels = pd.concat([df_labels, df_tmp])

df_labels = df_labels.sort_values("clustered_labels")

for i, label in enumerate(df_labels["clustered_labels"]):
    df_labels.iloc[i, 1] = f"cluster{label}"

palette = {'isogi': 'tab:blue', 'yukkuri': 'tab:orange', 'omoiyari': 'tab:green'}
ax = sns.histplot(data=df_labels, 
                  x="clustered_labels", 
                  hue="true_labels", 
                  multiple="dodge", 
                  palette=palette,
                  alpha=0.65,
                  shrink=0.5)
sns.set(rc={'figure.figsize':(10, 6)})
ax.set(title=dist)
plt.show()

time_np = to_time_series_dataset(df_clustering.T)

clus0, clus1, clus2 = [], [], []
for i, j in zip(df_labels["clustered_labels"], df_labels["true_labels"]):
    if i == 0:
        clus0.append(j)
    elif i == 1:
        clus1.append(j)
    else:
        clus2.append(j)

print(f'clus0({len(clus0)}): {Counter(clus0)}')
print(f'clus1({len(clus1)}): {Counter(clus1)}')
print(f'clus2({len(clus2)}): {Counter(clus2)}')

mf.plot_result_of_clustering(time_np, labels_euclidean, n_clusters)
mf.find_proper_num_clusters(df_clustering)

# %% funcs
def calculate_accuracy_of_clustering(clus0, clus1, clus2):
    a = Counter(clus0)
    b = Counter(clus1)
    c = Counter(clus2)
    total_urg = a["isogi"] + b["isogi"] + c["isogi"]
    tp = a["isogi"]
    fp = b["isogi"] + c["isogi"]
    fn = a["yukkuri"] + a["omoiyari"]
    tn = b["yukkuri"] + b["omoiyari"] + c["yukkuri"] + c["omoiyari"]
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    print(accuracy)

# %% perform clusterings for each condition and hopefully find specific patterns for that condition
def make_df_for_clustering_per_conditiosn(cond):
    df_cond = pd.DataFrame()
    for ID in tqdm(SUBJECTS):
        for trial in TRIALS:
            df_cond_tmp = mf.make_df_trial(df_all, ID, cond, 20, trial)
            df_cond_tmp = df_cond_tmp["dist_from_start"]
            df_cond = pd.concat([df_cond, df_cond_tmp], axis=1)
    
    return df_cond
        
df_omoi = make_df_for_clustering_per_conditiosn("omoiyari")
df_isogi = make_df_for_clustering_per_conditiosn("urgent")
df_yukkuri = make_df_for_clustering_per_conditiosn("nonurgent")

def tsclustering(df, n_clusters):
    km_euclidean = TimeSeriesKMeans(n_clusters=n_clusters, metric='dtw', random_state=2)
    labels_euclidean = km_euclidean.fit_predict(df["dist_from_start"].T)
    time_np = time_np = to_time_series_dataset(df.T)
    mf.plot_result_of_clustering(time_np, labels_euclidean, n_clusters)

mf.find_proper_num_clusters(df_omoi)
tsclustering(df_omoi, 3)

mf.find_proper_num_clusters(df_isogi)
tsclustering(df_isogi, 3)

mf.find_proper_num_clusters(df_yukkuri)
tsclustering(df_yukkuri, 3)

# %% why doesn't it calculate degrees

# need to consider how to deal with unmoving degree.
# when the position of time t and time t+1 is same, the degree will be None but 
# it should be punished most, but how?