# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from sklearn.cluster import KMeans
import warnings
# warnings.simplefilter('ignore')
from sklearn import metrics
from sklearn import svm
from sklearn.preprocessing import StandardScaler

from gc import collect as g
from collections import Counter
import math
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
df_clustering = mf.make_df_for_clustering(df_all, 22, 20, dist)

true_labels = ["omoiyari", "isogi", "yukkuri"] * 29 * 8
# for personal clustering
true_labels = ["omoiyari", "isogi", "yukkuri"] * 8


df_clustering = pd.DataFrame()
for ID in tqdm(SUBJECTS):
    df = mf.make_df_for_clustering(df_all, ID, 20, dist)
    df_clustering = pd.concat([df_clustering, df], axis=1)


scaler_std = StandardScaler()
df_clustering = scaler_std.fit_transform(df_clustering)


time_np = to_time_series_dataset(df_clustering.T)
n = 3
start = time.time()
km_euclidean = TimeSeriesKMeans(n_clusters=n, metric='dtw', verbose=1, random_state=2)
labels_euclidean = km_euclidean.fit_predict(df_clustering.T)
print(labels_euclidean)
end = time.time()
print(np.round(end-start, 1), "seconds")

df_labels = pd.DataFrame({"true_labels": true_labels,
                          "clustered_labels": labels_euclidean})


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


fig = plt.figure(figsize=(12, 4))
for i in range(n):
    ax = fig.add_subplot(1, 3, i+1)
    clus_arr = time_np[labels_euclidean == i]
    for x in clus_arr:
        ax.plot(x.ravel(), 'k-', alpha=0.2)
    ax.plot(km_euclidean.cluster_centers_[i].ravel(), 'r-')
    datanum = np.count_nonzero(labels_euclidean == i)
    ax.text(0.5, max(clus_arr[1])*0.8, f'Cluster{i} : n = {datanum}')
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


# %% calculate degrees
df_trial = mf.make_df_trial(df_all, ID=1, cond="omoiyari", agents=20, trial=1)
posXtplus1 = pd.concat([df_trial.posX[1:], pd.Series([None])], ignore_index=True)
posYtplus1 = pd.concat([df_trial.posY[1:], pd.Series([None])], ignore_index=True)
df_trial['posXt+1'] = posXtplus1
df_trial['posYt+1'] = posYtplus1

def calc_deg(x0, y0, x1, y1, x2=880, y2=880):
    try:
        vec1 = [x1 - x0, y1 - y0]
        vec2 = [x2 - x0, y2 - y0]
      
        absvec1 = np.linalg.norm(vec1)
        absvec2 = np.linalg.norm(vec2)
        inner = np.inner(vec1, vec2)
        cos_theta = inner / (absvec1 * absvec2)
        theta = math.degrees(math.acos(cos_theta))
        
        return theta
    
    except TypeError:
        return None
    
# might need to consider how to deal with unmoving degree.
# when the position of time t and time t+1 is same, the degree will be None but 
# it should be punished most.

# deg = df_trial.apply(lambda df: calc_deg(
#     df["posX"], df["posY"], df["posXt+1"], df["posYt+1"]
#     ), axis=1)


# %% run classification for all participants' data
cols_to_classify = ['completion_time', 'dist_top12_closest', 'dist_actual_ideal', 'condition']
df_sum = pd.DataFrame(columns=cols_to_classify)

for cond in CONDITIONS:
    for ID in SUBJECTS:
        for trial in TRIALS:
            df_trial = mf.make_df_trial(df_all, ID, cond, agents, trial)
            # if cond == 'urgent': 
            #     condition = 'urgent'
            # else:
            #     condition = 'non-urgent'
            tmp = pd.DataFrame(
                {'completion_time': df_trial['timerTrial'].iloc[-1],
                 'dist_top12_closest': np.mean(df_trial['dist_top12_closest']),
                 'dist_actual_ideal': np.mean(df_trial['dist_actual_ideal']),
                 'condition': cond},
                index=[f'ID{ID}_{cond}_agents{agents}_trial{trial}']
            )
            df_sum = pd.concat([df_sum, tmp], ignore_index=True)

df_sum = pd.DataFrame(columns=cols_to_classify)
for cond in CONDITIONS:
    for trial in TRIALS:
        df_tri = mf.make_df_trial(df_all, 3, cond, 20, trial)
        df_class = pd.DataFrame(
            {'completion_time': df_trial['timerTrial'].iloc[-1],
             'dist_top12_closest': np.mean(df_trial['dist_top12_closest']),
             'dist_actual_ideal': np.mean(df_trial['dist_actual_ideal']),
             'condition': cond},
            index=[f'ID{ID}_{cond}_agents{agents}_trial{trial}']
        )
        df_sum = pd.concat([df_sum, df_class])

X = df_sum[['completion_time', 'dist_top12_closest', 'dist_actual_ideal']]
Y = df_sum['condition']
x_train, y_train = X[:400], Y[:400]
x_test, y_test = X[400:], Y[400:]

#Create a svm Classifier
clf = svm.SVC(kernel='rbf') # Linear Kernel
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
acc_score = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", np.round(acc_score, 4)*100, "%")

kmeans_model = KMeans(n_clusters=3)
clusters = kmeans_model.fit_predict(X)
len_per_cond = int(len(clusters) / len(CONDITIONS))

label_urg = clusters[:len_per_cond]
label_nonurg = clusters[len_per_cond:len_per_cond*2]
label_omoi = clusters[len_per_cond*2:]

label_nonurg_omoi = clusters[len_per_cond:]

a = Counter(label_urg)
b = Counter(label_nonurg)
c = Counter(label_omoi)
total_urg = a[1] + b[1] + c[1]
tp = a[1]
fp = b[1] + c[1]
fn = a[0]
tn = b[0] + b[2] + c[0] + c[2]
accuracy = (tp + tn) / (tp + fp + fn + tn)
print(accuracy)

print(f'urgent({len(label_urg)}) {Counter(label_urg)}')
print(f'nonurgent({len(label_nonurg)}) {Counter(label_nonurg)}')
print(f'omoiyari({len(label_omoi)}) {Counter(label_omoi)}')

print(f'non-urgent({len(label_nonurg_omoi)}) {Counter(label_nonurg_omoi)}')

