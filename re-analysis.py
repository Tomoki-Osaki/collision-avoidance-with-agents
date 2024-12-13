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

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tslearn.neighbors import KNeighborsTimeSeriesClassifier as tsKNTSC

import warnings
warnings.simplefilter('ignore')
from collections import Counter
from tqdm import tqdm
from gc import collect as g
import myfuncs as mf
from myfuncs import col, SUBJECTS, CONDITIONS, TRIALS

#%% loading the data
df_all = mf.make_dict_of_all_info(SUBJECTS)

# %% try to calculate the braking rate
for cond in CONDITIONS:
    tmp = mf.make_df_trial(df_all, 7, cond, 20, 1)
    
    brakings = tmp.loc[:, tmp.columns.str.startswith('other')]
    myinfo = tmp.loc[:, tmp.columns.str.startswith('my')]
    brakings = pd.concat([myinfo, brakings], axis=1)    
    
    max_brake = []
    for data in brakings.iterrows():
        brakeRates = []
        for i in range(1, 21):
            brake = mf.V_BrakingRate(
                data[1]['myMoveX'], data[1]['myMoveY'],
                data[1]['myNextX'], data[1]['myNextY'],
                data[1][f'other{i}MoveX'], data[1][f'other{i}MoveY'],
                data[1][f'other{i}NextX'], data[1][f'other{i}NextY'],
                return_when_undefined=-np.inf
            )
            brakeRates.append(brake)
        max_brake.append(max(brakeRates))

    tmp['BrakeRate'] = max_brake
    if cond == 'omoiyari': color='tab:green'
    elif cond == 'isogi': color='tab:blue'
    elif cond == 'yukkuri': color='tab:orange'
    plt.plot(max_brake)
plt.show()

# %%
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()

for data in tmp.iterrows():
    ax.scatter(data[1]['myNextX'], data[1]['myNextY'], color='blue')
plt.show()

fig, ax = plt.subplots()
def update(data):
    ax.cla()
    ax.scatter(data[1]['myNextX'], data[1]['myNextY'], color='blue')
    ax.vlines(x=data[1]['goalX1'], ymin=data[1]['goalY1'], ymax=data[1]['goalY2'], 
              color='gray', alpha=0.5, linestyles='dashed')
    ax.vlines(x=data[1]['goalX2'], ymin=data[1]['goalY1'], ymax=data[1]['goalY2'],
              color='gray', alpha=0.5, linestyles='dashed')
    ax.hlines(y=data[1]['goalY1'], xmin=data[1]['goalX1'], xmax=data[1]['goalX2'],
              color='gray', alpha=0.5, linestyles='dashed')
    ax.hlines(y=data[1]['goalY2'], xmin=data[1]['goalX1'], xmax=data[1]['goalX2'],
              color='gray', alpha=0.5, linestyles='dashed')
    
    for i in range(1, 21):
        ax.scatter(data[1][f'other{i}NextX'], data[1][f'other{i}NextY'], color='gray')
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    
anim = FuncAnimation(fig, update, frames=tmp.iterrows(), repeat=False, 
                     interval=200, cache_frame_data=False)
anim.save('video.mp4')

# %% perform clusterings for each condition and hopefully find specific patterns for that condition
def make_df_for_clustering_per_conditions(cond, feature):
    df_cond = pd.DataFrame()
    for ID in tqdm(SUBJECTS):
        for trial in TRIALS:
            df_cond_tmp = mf.make_df_trial(df_all, ID, cond, 20, trial)
            df_cond_tmp = df_cond_tmp[feature]
            df_cond = pd.concat([df_cond, df_cond_tmp], axis=1)
            
    if feature == 'dist_actual_ideal':
        df_cond = df_cond.iloc[1:]
    
    return df_cond
        
dist = 'dist_from_start'
cond = 'isogi'
df = make_df_for_clustering_per_conditions(cond, dist)

#mf.find_proper_num_clusters(df)
def clustering(df, cond, feature):
    km_euclidean = TimeSeriesKMeans(n_clusters=n, metric='dtw', random_state=2)
    labels_euclidean = km_euclidean.fit_predict(df[feature].T)
    df_res = df.copy().T
    #df_res['clustered'] = labels_euclidean.astype(int)
    df_res['clustered'] = labels_euclidean
    df_res['condition'] = cond
    return df_res

n = 3
df_res = clustering(df, cond, dist)

fig, axs = plt.subplots(1, n, figsize=(15, 5), sharex=True, sharey=True)
for data in df_res.iterrows():
    if data[1]['condition'] == 'omoiyari': color='tab:green'
    elif data[1]['condition'] == 'isogi': color='tab:blue'
    elif data[1]['condition'] == 'yukkuri': color='tab:orange'
    axs[data[1]['clustered']].plot(data[1][:-2], color=color, alpha=0.5)
for i in range(n):
    axs[i].grid()
plt.tight_layout()
plt.show()
print('\ndist:', dist); print('cond:', cond)

dist = 'dist_from_start'
df_res = pd.DataFrame()
for cond in CONDITIONS:
    df_tmp = make_df_for_clustering_per_conditions(cond, dist)
    df_tmp = clustering(df_tmp, cond, dist)
    df_res = pd.concat([df_res, df_tmp])
clus_col = df_res.pop('clustered')
cond_col = df_res.pop('condition')
df_res['clustered'] = clus_col.astype(int)
df_res['condition'] = cond_col

xmax, ymax = (180, 1000) if dist == 'dist_actual_ideal' else (180, 1300)

fig, axs = plt.subplots(3, 3, figsize=(10, 10), sharex=True, sharey=True)
for data in df_res.iterrows():
    if data[1]['condition'] == 'isogi': 
        n = 0
        color='tab:blue'
    elif data[1]['condition'] == 'yukkuri': 
        n = 1
        color='tab:orange'
    elif data[1]['condition'] == 'omoiyari': 
        n = 2
        color='tab:green'
    axs[n, data[1]['clustered']].plot(data[1][:-2], color=color, alpha=0.5)
    if data[1]['clustered'] == 1:
        axs[n, data[1]['clustered']].set_title(data[1]['condition'])
for i in range(3):
    for j in range(3):
        axs[i, j].grid()
plt.show()

    
# %% plot all clusterings in one figure
df_res = clustering(df, dist)
fig, ax = plt.subplots(1, n, figsize=(15, 5), sharex=True, sharey=True)
for data in df_res.iterrows():
    if data[1]['clustered'] == 0: color='tab:green'
    elif data[1]['clustered'] == 1: color='tab:blue'
    elif data[1]['clustered'] == 2: color='tab:orange'
    ax[int(data[1]['clustered'])].set_xlim(0, xmax)
    ax[int(data[1]['clustered'])].set_ylim(0, ymax)
    ax[int(data[1]['clustered'])].plot(data[1][:-1], color=color, alpha=0.5)
for i in range(n):
    ax[i].grid()
plt.tight_layout()

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

# look through all subjects' clustering patterns
# dist = 'dist_closest'
# for ID in SUBJECTS:
#     df = pd.DataFrame()
#     for trial in TRIALS:
#         for cond in CONDITIONS: # ['isogi', 'yukkuri', 'omoiyari']
#             tmp = mf.make_df_trial(df_all, ID, cond, 20, trial)[dist]
#             if dist == 'dist_actual_ideal':
#                 tmp = tmp.iloc[1:]
#             df = pd.concat([df, tmp], axis=1)
    
#     n = 3
#     km_euclidean = TimeSeriesKMeans(n_clusters=n, metric='dtw', random_state=2)
#     labels_euclidean = km_euclidean.fit_predict(df.T)
#     print(Counter(labels_euclidean))
#     time_np = to_time_series_dataset(df.T)
#     true_labs = CONDITIONS * 8
#     colors =  ['tab:blue', 'tab:orange', 'tab:green']
    
#     res_df = df.T.copy()
#     res_df['clustered'] = labels_euclidean
#     res_df['true_labels'] = true_labs
    
#     fig, ax = plt.subplots(1, n, figsize=(15, 5), sharex=True, sharey=True)
#     for idx, data in enumerate(res_df.iterrows()):
#         if data[1]['true_labels'] == 'omoiyari': color = 'tab:green'
#         elif data[1]['true_labels'] == 'isogi': color='tab:blue'
#         elif data[1]['true_labels'] == 'yukkuri': color = 'tab:orange'
#         ax[data[1]['clustered']].plot(data[1][:-2], color=color, alpha=0.7)
#     plt.tight_layout()
#     plt.show()
#     print('ID', ID)

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

# because clusters for each subject are not related, this way of counting is 
# not proper and must need to consider anothey way!!!
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

# %% why doesn't it calculate degrees?

# need to consider how to deal with unmoving degree.
# when the position of time t and time t+1 is same, the degree will be None but 
# it should be punished most, but how?
