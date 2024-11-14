# %% import libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tslearn.neighbors import KNeighborsTimeSeriesClassifier as tsKNTSC

import warnings
warnings.simplefilter('ignore')
from collections import Counter
from tqdm import tqdm
import myfuncs as mf

# %% define global variables
SUBJECTS = mf.SUBJECTS
CONDITIONS = mf.CONDITIONS # [urgent, nonurgent, omoiyari]
AGENTS = mf.AGENTS
NUM_AGENTS = mf.NUM_AGENTS
TRIALS = mf.TRIALS

#%% loading the data
df_all = mf.make_dict_of_all_info(SUBJECTS)

# %% plot data
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

# %% try standardize
df = mf.make_df_for_clustering(df_all, 1, 20, 'dist_actual_ideal')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_std = scaler.fit_transform(df)

features = ['timerTrial', 'dist_from_start', 'dist_actual_ideal',
            'dist_top12_closest']
arr = np.zeros((696, len(features), 44))
i = 0
for ID in SUBJECTS:
    for trial in TRIALS:
        for cond in CONDITIONS:
            df_tri = mf.make_df_trial(df_all, ID, cond, 20, trial)[:44]
            df_arr = np.array(df_tri[features])
            arr[i] += df_arr.T
            i += 1

ylabs = np.array(['urgent', 'nonurgent', 'omoiyari'] * 232)
ylabs = np.array(['urgent', 'not-urgent', 'not-urgent'] * 232)
ylabs = np.array(['not-nonurgent', 'nonurgent', 'not-nonurgent'] * 232)
ylabs = np.array(['not-omoiyari', 'not-omoiyari', 'omoiyari'] * 232)

X_train, X_test, y_train, y_test = train_test_split(arr, ylabs, test_size=0.2)

clf = tsKNTSC(n_neighbors=1, weights='distance', metric='dtw')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)

# cannot distinguish not-nonurgent and nonurgent but can distinguish not-omoiyari 
# and not-omoiyari

# %% time series clustering
dist = "dist_actual_ideal"
dist = "dist_from_start"
dist = "dist_closest"
dist = "dist_top12_closest"
n_clusters = 3

true_labels = ["omoiyari", "isogi", "yukkuri"] * 8

dfs = pd.DataFrame()
for ID in SUBJECTS:
    for cond in CONDITIONS:
        for trial in TRIALS:
            df_tmp = df_all[f'ID{ID}'][cond]['agents20'][f'trial{trial}']['timerTrial']
            dfs[f'ID{ID}cond{cond}tri{trial}timer'] = df_tmp**2
        
        
true_labels = ["urgent", "nonurgent", "omoiyari"] * 8        
df = pd.DataFrame()
for tri in TRIALS:
    for cond in CONDITIONS:
        df_tmp = df_all['ID1'][cond]['agents20'][f'trial{tri}']['timerTrial']
        df_tmp = pd.Series(df_tmp, name=f'cond_{cond}_tri{tri}')
        df = pd.concat([df, df_tmp], axis=1)

km_euclidean = TimeSeriesKMeans(n_clusters=n_clusters, metric='dtw', random_state=2)
labels_euclidean = km_euclidean.fit_predict(df.T)        
        
df_comp = pd.DataFrame({"true_labels": true_labels,
                        "clustered_labels": labels_euclidean})

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

# %% try downsampling and confirm how much information was lost
from sklearn.metrics import accuracy_score
from sktime.classification.kernel_based import RocketClassifier

df = mf.make_df_for_clustering(df_all, 1, 20, 'dist_from_start')
true_labels = np.array(['omoiyari', 'urgent', 'nonurgent'] * 8)

rocket = RocketClassifier(num_kernels=2000)
rocket.fit(df, true_labels)
y_pred = rocket.predict(df)

accuracy_score(true_labels, y_pred)

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
def make_df_for_clustering_per_conditions(cond):
    df_cond = pd.DataFrame()
    for ID in tqdm(SUBJECTS):
        for trial in TRIALS:
            df_cond_tmp = mf.make_df_trial(df_all, ID, cond, 20, trial)
            df_cond_tmp = df_cond_tmp["dist_from_start"]
            df_cond = pd.concat([df_cond, df_cond_tmp], axis=1)
    
    return df_cond
        
df_omoi = make_df_for_clustering_per_conditions("omoiyari")
df_isogi = make_df_for_clustering_per_conditions("urgent")
df_yukkuri = make_df_for_clustering_per_conditions("nonurgent")

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
