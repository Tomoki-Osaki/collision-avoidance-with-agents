# %% import libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
# sns.set_theme()
# sns.reset_orig()

from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tslearn.neighbors import KNeighborsTimeSeriesClassifier as tsKNTSC

import warnings
warnings.simplefilter('ignore')
from collections import Counter
from tqdm import tqdm
from gc import collect as g
import myfuncs as mf
from myfuncs import col, SUBJECTS, CONDITIONS, AGENTS, NUM_AGENTS, TRIALS

#%% loading the data
df_all = mf.make_dict_of_all_info(SUBJECTS)

# %% try to calculate the braking rate
tmp = mf.make_df_trial(df_all, 1, 'isogi', 20, 1)
mf.V_BrakingRate(tmp['myMoveX'], tmp['myMoveY'], 
                 tmp['myNextX'], tmp['myNextY'], 
                 tmp['other1MoveX'], tmp['other1MoveY'], 
                 tmp['other1NextX'], tmp['other1NextY'])

braking = tmp.apply(lambda df: mf.V_BrakingRate(
    df['myMoveX'], df['myMoveY'], 
    df['myNextX'], df['myNextY'], 
    df['other2MoveX'], df['other2MoveY'], 
    df['other2NextX'], df['other2NextY']
    ), axis=1
)

# %% time series clustering 
# per condition and look through what kind of patterns could be found
ID = 27

dist = 'dist_actual_ideal'
dist = 'dist_from_start'
df = pd.DataFrame()
for trial in TRIALS:
    for cond in CONDITIONS: # ['isogi', 'yukkuri', 'omoiyari']
        tmp = mf.make_df_trial(df_all, ID, cond, 20, trial)[dist]
        if dist == 'dist_actual_ideal':
            tmp = tmp.iloc[1:]
        df = pd.concat([df, tmp], axis=1)

# df = df.filter(like='yukkuri')

for ID in SUBJECTS:
    mf.plot_dist_compare_conds(df_all, ID, 20, dist)
mf.plot_dist_per_cond(df_all, ID, 20, "dist_actual_ideal")

mf.find_proper_num_clusters(df)

n = 3
km_euclidean = TimeSeriesKMeans(n_clusters=n, metric='dtw', random_state=2)
labels_euclidean = km_euclidean.fit_predict(df.T)
print(Counter(labels_euclidean))
time_np = to_time_series_dataset(df.T)
true_labs = CONDITIONS * 8
colors =  ['tab:blue', 'tab:orange', 'tab:green']

res_df = df.T.copy()
res_df['clustered'] = labels_euclidean
res_df['true_labels'] = true_labs

clus0 = Counter(res_df.query('clustered == 0')['true_labels'])
clus1 = Counter(res_df.query('clustered == 1')['true_labels'])
clus2 = Counter(res_df.query('clustered == 2')['true_labels'])
print('clus0:', clus0)
print('clus1:', clus1)
print('clus2:', clus2)

fig, ax = plt.subplots(1, n, figsize=(15, 5), sharex=True, sharey=True)
for idx, data in enumerate(res_df.iterrows()):
    if data[1]['true_labels'] == 'omoiyari': color = 'tab:green'
    elif data[1]['true_labels'] == 'isogi': color='tab:blue'
    elif data[1]['true_labels'] == 'yukkuri': color = 'tab:orange'
    ax[data[1]['clustered']].plot(data[1][:-2], color=color, alpha=0.7)
plt.tight_layout()
plt.show()


dist = 'dist_closest'
for ID in SUBJECTS:
    df = pd.DataFrame()
    for trial in TRIALS:
        for cond in CONDITIONS: # ['isogi', 'yukkuri', 'omoiyari']
            tmp = mf.make_df_trial(df_all, ID, cond, 20, trial)[dist]
            if dist == 'dist_actual_ideal':
                tmp = tmp.iloc[1:]
            df = pd.concat([df, tmp], axis=1)
    
    n = 3
    km_euclidean = TimeSeriesKMeans(n_clusters=n, metric='dtw', random_state=2)
    labels_euclidean = km_euclidean.fit_predict(df.T)
    print(Counter(labels_euclidean))
    time_np = to_time_series_dataset(df.T)
    true_labs = CONDITIONS * 8
    colors =  ['tab:blue', 'tab:orange', 'tab:green']
    
    res_df = df.T.copy()
    res_df['clustered'] = labels_euclidean
    res_df['true_labels'] = true_labs
    
    fig, ax = plt.subplots(1, n, figsize=(15, 5), sharex=True, sharey=True)
    for idx, data in enumerate(res_df.iterrows()):
        if data[1]['true_labels'] == 'omoiyari': color = 'tab:green'
        elif data[1]['true_labels'] == 'isogi': color='tab:blue'
        elif data[1]['true_labels'] == 'yukkuri': color = 'tab:orange'
        ax[data[1]['clustered']].plot(data[1][:-2], color=color, alpha=0.7)
    plt.tight_layout()
    plt.show()
    print('ID', ID)
    

# %% KNeighborsTimeSeriesClassifier from tslean (classification)
# ex. shape (40, 100, 6) = (data, timepoints, variables)
# each timepoint has 6 variables
# thus, the prepared data's shape must be (696, 191, ?5)
# where, 696=3x8x29, 191=max_length, ?5=len(features)

# be careful not to make tuples
features = ['timerTrial', 
            'dist_from_start', 
            'dist_actual_ideal',
            'dist_closest']
            #'dist_top12_closest']

conds = ['isogi', 'yukkuri', 'omoiyari']
ylabs = np.array(['isogi', 'yukkuri', 'omoiyari'] * 232)

conds = ['yukkuri', 'omoiyari']
ylabs = np.array(['yukkuri', 'omoiyari'] * 232)

conds = ['isogi', 'yukkuri']
ylabs = np.array(['isogi', 'yukkuri'] * 232)

conds = ['isogi', 'omoiyari']
ylabs = np.array(['isogi', 'omoyari'] * 232)

arr = mf.make_arr_for_train_test(df_all, features, conds, ylabs)

clf = tsKNTSC(n_neighbors=1, weights='uniform', metric='dtw')

X_train, X_test, y_train, y_test = train_test_split(arr, ylabs, test_size=0.25)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
summary = pd.DataFrame(
    classification_report(y_test, y_pred, output_dict=True)
)
print(summary.T)

repeat = 6
for i in tqdm(range(repeat)):
    X_train, X_test, y_train, y_test = train_test_split(arr, ylabs, test_size=0.25)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    if i == 0:
        summary = pd.DataFrame(
            classification_report(y_test, y_pred, output_dict=True)
        )
    else:
        summary += pd.DataFrame(
            classification_report(y_test, y_pred, output_dict=True)
        )
summary /= repeat

print(summary.T)
print(features)
print(conds)

# %% plot data
ID = 20
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
        
true_labels = ["isogi", "yukkuri", "omoiyari"] * 8        
df = pd.DataFrame()
for tri in TRIALS:
    for cond in CONDITIONS:
        df_tmp = df_all['ID1'][cond]['agents20'][f'trial{tri}'][dist]
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

# %% perform clusterings for each condition and hopefully find specific patterns for that condition
def make_df_for_clustering_per_conditions(cond, feature):
    df_cond = pd.DataFrame()
    for ID in tqdm(SUBJECTS):
        for trial in TRIALS:
            df_cond_tmp = mf.make_df_trial(df_all, ID, cond, 20, trial)
            df_cond_tmp = df_cond_tmp[feature]
            df_cond = pd.concat([df_cond, df_cond_tmp], axis=1)
    
    return df_cond
        
df_omoi = make_df_for_clustering_per_conditions("omoiyari", 'dist_actual_ideal')
df_isogi = make_df_for_clustering_per_conditions("isogi", 'dist_from_start')
df_yukkuri = make_df_for_clustering_per_conditions("yukkuri", 'dist_from_start')

def tsclustering(df, n_clusters, feature):
    km_euclidean = TimeSeriesKMeans(n_clusters=n_clusters, metric='dtw', random_state=2)
    labels_euclidean = km_euclidean.fit_predict(df[feature].T)
    time_np = to_time_series_dataset(df.T)
    mf.plot_result_of_clustering(km_euclidean, time_np, labels_euclidean, n_clusters)

n = 3
mf.find_proper_num_clusters(df_omoi)
tsclustering(df_omoi, n, 'dist_actual_ideal')

mf.find_proper_num_clusters(df_isogi)
tsclustering(df_isogi, n, 'dist_from_start')

mf.find_proper_num_clusters(df_yukkuri)
tsclustering(df_yukkuri, n, 'dist_from_start')
    

# %% why doesn't it calculate degrees

# need to consider how to deal with unmoving degree.
# when the position of time t and time t+1 is same, the degree will be None but 
# it should be punished most, but how?
