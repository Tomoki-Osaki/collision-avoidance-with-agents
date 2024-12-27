# %% import libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
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
#df_all = mf.make_dict_of_all_info(SUBJECTS)
df_all = mf.make_dict_of_all_info([1, 2, 3, 4, 5])

# %% practice of implement of the awareness model
df_ori = mf.make_df_trial(df_all, 5, 'isogi', 20, 1)
#df = df[20:50]
#mf.anim_movements(df)
df = df_ori.iloc[25, :]
df2 = df_ori.iloc[24, :]

def plot_pos(df):
    plt.scatter(df['myNextX'], df['myNextY'], color='blue')
    for i in range(1, 21):
        plt.scatter(df[f'other{i}NextX'], df[f'other{i}NextY'], color='gray')
        plt.annotate(i, xy=(df[f'other{i}NextX'], df[f'other{i}NextY']))

plot_pos(df)

def calc_nic(df, agent):
    my_pos = (df['myNextX'], df['myNextY'])
    agent_pos = (df[f'other{agent}NextX'], df[f'other{agent}NextY'])
    cp = ( (my_pos[0] + agent_pos[0]) / 2, (my_pos[1] + agent_pos[1]) / 2 )
    dist_cp_me = mf.calc_distance(cp[0], cp[1], my_pos[0], my_pos[1])
    
    Nic_agents = []
    for i in range(1, 21):
        other_pos = (df[f'other{i}NextX'], df[f'other{i}NextY'])
        dist_cp_other = mf.calc_distance(cp[0], cp[1], other_pos[0], other_pos[1])
        if dist_cp_other <= dist_cp_me and not i == agent:
            Nic_agents.append(i)
    
    return Nic_agents
    
for agent in range(1, 21):
    Nic = len(calc_nic(df, agent))
    posx1, posy1 = (df['myNextX'], df['myNextY'])
    velx1, vely1 = (df['myNextX'], df['myNextY'])
    posx_tminus1, posy_tminus1 = (df2['myNextX'], df2['myNextY'])
    posx2, posy2 = (df[f'other{agent}NextX'], df[f'other{agent}NextY'])
    velx2, vely2 = (df[f'other{agent}NextX'], df[f'other{agent}NextY'])
    Px = posx2 - posx1
    Py = posy2 - posy1
    dist1 = mf.calc_distance(df2['myMoveX'], df2['myMoveY'], 
                             df['myMoveX'], df['myMoveY'])
    Vself = dist1
    dist2 = mf.calc_distance(df2[f'other{agent}MoveX'], df2[f'other{agent}MoveY'], 
                             df[f'other{agent}MoveX'], df[f'other{agent}MoveY'])
    Vother = dist2
    
    slope1 = (posy1 - posy_tminus1) / (posx1 - posx_tminus1)
    slope2 = (posy2 - posy1) / (posx2 - posx1)
    theta = np.arctan(np.abs(slope1 - slope2) / (1 + slope1 * slope2))
    
    deltaTTCP = mf.deltaTTCP_N(velx1, vely1, posx1, posy1, velx2, vely2, posx2, posy2)
    awm = mf.awareness_model(deltaTTCP, Px, Py, Vself, Vother, theta, Nic)  
    print('\ndeltaTTCP, Px, Py, Vself, Vother, theta, Nic')
    print(deltaTTCP, Px, Py, Vself, Vother, theta, Nic)
    print(agent, awm)

# %% try to calculate the Nic and complete the awareness model
# must rescale the values. The parameters of Awareness model might be calculated 
# using the different sclaes such as meter, m/s, and so on.
df = mf.make_df_trial(df_all, 10, 'isogi', 20, 6)
col(df)

tminus1 = ...
focused_other = [0]
others_in_view = [i for i in range(1, 21)]
for data in df.iterrows():
    awm = []
    if not data[0] == 0:
        for other in range(1, 21):
            velx1, vely1 = data[1]['myMoveX'], data[1]['myMoveY']
            posx1, posy1 = data[1]['myNextX'], data[1]['myNextY']
            velx2, vely2 = data[1][f'other{other}MoveX'], data[1][f'other{other}MoveY']
            posx2, posy2 = data[1][f'other{other}NextX'], data[1][f'other{other}NextY']
            
            posx_tminus1 = tminus1[1]['myNextX']
            posy_tminus1 = tminus1[1]['myNextY']
            
            deltaTTCP = mf.deltaTTCP_N(velx1, vely1, posx1, posy1, 
                                       velx2, vely2, posx2, posy2)
            Px = posx2 - posx1
            Py = posy2 - posy1
            
            try:
                slope1 = (posy1 - posy_tminus1) / (posx1 - posx_tminus1)
                slope2 = (posy2 - posy1) / (posx2 - posx1)
                theta = np.arctan(np.abs(slope1 - slope2) / (1 + slope1 * slope2))
                if np.rad2deg(theta) > 90:
                    others_in_view.pop(other+1)
            except ZeroDivisionError:
                awm.append((other, 0))
                continue
            
            Nic = -1
            dist_to_the_other = data[1][f'distOther{other}']
            for i in others_in_view:
                if data[1][f'distOther{i}'] <= dist_to_the_other:
                    Nic += 1
            
            dist1 = mf.calc_distance(
                data[1]['myNextX'], data[1]['myNextY'], 
                tminus1[1]['myNextX'], tminus1[1]['myNextY']
            )
            speed1 = dist1 / 100
            dist2 = mf.calc_distance(
                data[1][f'other{other}NextX'], data[1][f'other{other}NextY'], 
                tminus1[1][f'other{other}NextX'], tminus1[1][f'other{other}NextY']
            )
            speed2 = dist2 / 100
            
            aw = mf.awareness_model(deltaTTCP, Px, Py, speed1, speed2, theta, Nic)
            if other in others_in_view:
                awm.append((other, aw))
            
        to_focus = [i for i, j in awm if j == 1.0]
        dist_other = [(other, data[1][f'distOther{other}']) for other in to_focus]
        dist_other.sort(key=lambda tup: tup[1])
        try:
            focused = dist_other[0][0]
            br_posx2 = data[1][f'other{focused}NextX']
            br_posy2 = data[1][f'other{focused}NextY']
            br_velx2 = data[1][f'other{focused}MoveX']
            br_vely2 = data[1][f'other{focused}MoveY']
            braking_rate_focused = mf.BrakingRate(
                velx1, vely1, posx1, posy1, 
                br_velx2, br_vely2, br_posx2, br_posy2
            )
        except IndexError:
            focused = 0
            braking_rate_focused = 0
            
        focused_other.append(focused)
        
    tminus1 = data

df['focused_other'] = focused_other

# %% try to animate how the focused others were selected
fig, ax = plt.subplots()
tminus1 = ...
def update(data):
    ax.cla()
    ax.scatter(data[1]['myNextX'], data[1]['myNextY'], color='blue')
    for i in range(1, 21):
        ax.scatter(data[1][f'other{i}NextX'], data[1][f'other{i}NextY'], color='gray')
        
    ax.vlines(x=data[1]['goalX1'], ymin=data[1]['goalY1'], ymax=data[1]['goalY2'], 
              color='gray', alpha=0.5)
    ax.vlines(x=data[1]['goalX2'], ymin=data[1]['goalY1'], ymax=data[1]['goalY2'],
              color='gray', alpha=0.5)
    ax.hlines(y=data[1]['goalY1'], xmin=data[1]['goalX1'], xmax=data[1]['goalX2'],
              color='gray', alpha=0.5)
    ax.hlines(y=data[1]['goalY2'], xmin=data[1]['goalX1'], xmax=data[1]['goalX2'],
              color='gray', alpha=0.5)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.tick_params(left=False, right=False, labelleft=False, 
                   labelbottom=False, bottom=False) 
    
    focused = data[1]['focused_other']
    if not focused == 0:
        # ax.plot((data[1]['myNextX'], data[1][f'other{focused}NextX']),
        #         (data[1]['myNextY'], data[1][f'other{focused}NextY']), 
        #         color='red')
        ax.annotate('', 
                    xy=(data[1][f'other{focused}NextX'], 
                        data[1][f'other{focused}NextY']), 
                    xytext=(data[1]['myNextX'],
                            data[1]['myNextY']),
                    arrowprops=dict(width=0.5, headwidth=5,
                                    facecolor='red', edgecolor='red'))
    
anim = FuncAnimation(fig, update, frames=df.iterrows(), repeat=False, 
                     interval=250, cache_frame_data=False)
anim.save('awareness.mp4')    



################################################################################
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
        
feature = 'dist_from_start'
feature = 'BrakeRate_Sum'
cond = 'isogi'
df = make_df_for_clustering_per_conditions(cond, feature)
df.fillna(0, inplace=True)

def clustering(df, n, cond, feature):
    km_euclidean = TimeSeriesKMeans(n_clusters=n, metric='dtw', random_state=2)
    labels_euclidean = km_euclidean.fit_predict(df[feature].T)
    df_res = df.copy().T
    #df_res['clustered'] = labels_euclidean.astype(int)
    df_res['clustered'] = labels_euclidean
    df_res['condition'] = cond
    return df_res

mf.find_proper_num_clusters(df)
n = 5
df_res = clustering(df, n, cond, feature)

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
print('\nfeature:', feature); print('cond:', cond)


# plot the results of all conditions in one figure (3x3)
feature = 'dist_from_start'
feature = 'dist_actual_ideal'
n = 3
df_res = pd.DataFrame()
for cond in CONDITIONS:
    df_tmp = make_df_for_clustering_per_conditions(cond, feature)
    df_tmp = clustering(df_tmp, n, cond, feature)
    df_res = pd.concat([df_res, df_tmp])
clus_col = df_res.pop('clustered')
cond_col = df_res.pop('condition')
df_res['clustered'] = clus_col.astype(int)
df_res['condition'] = cond_col

xmax, ymax = (180, 1000) if feature == 'dist_actual_ideal' else (180, 1300)

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
    # if data[1]['clustered'] == 1:
    #     axs[n, data[1]['clustered']].set_title(data[1]['condition'])
for i in range(3):
    for j in range(3):
        axs[i, j].grid()
#plt.tight_layout()
plt.show()

# %% plot all clusterings in one figure
df_res = clustering(df, feature)
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

feature = 'dist_actual_ideal'
feature= 'dist_from_start'
df = pd.DataFrame()
for trial in TRIALS:
    for cond in CONDITIONS: # ['isogi', 'yukkuri', 'omoiyari']
        tmp = mf.make_df_trial(df_all, ID, cond, 20, trial)[feature]
        if feature== 'dist_actual_ideal':
            tmp = tmp.iloc[1:]
        df = pd.concat([df, tmp], axis=1)

# df = df.filter(like='yukkuri')

mf.plot_dist_compare_conds(df_all, ID, 20, feature)
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

# %% KNeighborsTimeSeriesClassifier from tslean (classification)
# ex. shape (40, 100, 6) = (data, timepoints, variables)
# each timepoint has 6 variables
# thus, the prepared data's shape must be (696, 191, ?5)
# where, 696=3x8x29, 191=max_length, ?5=len(features)

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
feature= "dist_actual_ideal"
feature= "dist_from_start"
feature= "dist_closest"
feature= "dist_top12_closest"
n_clusters = 3
        
true_labels = ["isogi", "yukkuri", "omoiyari"] * 8        
df = pd.DataFrame()
for tri in TRIALS:
    for cond in CONDITIONS:
        df_tmp = df_all['ID1'][cond]['agents20'][f'trial{tri}'][feature]
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
    df_clustering = mf.make_df_for_clustering(df_all, ID, 20, feature)
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
ax.set(title=feature)
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
