import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from gc import collect as g
from typing import Literal
import math
from tqdm import tqdm
import myfuncs as mf

SUBJECTS = mf.SUBJECTS
CONDITIONS = mf.CONDITIONS # [urgent, nonurgent, omoiyari]
AGENTS = mf.AGENTS
NUM_AGENTS = mf.NUM_AGENTS
TRIALS = mf.TRIALS

df_all = mf.make_dict_of_all_info(SUBJECTS)

# %% make a dataframe for classification (supervised)
ID = 18
agents = 10
cond = "nonurgent"
trial = 3

df_trial = mf.make_df_trial(df_all, ID, cond, agents, trial)
cols = df_trial.columns


dist_closest = np.mean(df_trial['dist_closest'])
dist_actual_ideal = np.mean(df_trial['dist_actual_ideal'])
completion_time = df_trial['timerTrial'].iloc[-1]
tmp = pd.DataFrame({'completion_time': completion_time,
                    'dist_closest': dist_closest,
                    'dist_actual_ideal': dist_actual_ideal,
                    'condition': 'nonurgent'},
                   index=['ID14_nonurgent_agents20_trial3'])

tmp = df_trial[['dist_closest', 'dist_actual_ideal']]
tmp = tmp.apply(np.mean, axis=0)

# %% run classification
ID = 5
agents = 20
df_sum = pd.DataFrame(columns=[
    'completion_time', 'dist_closest', 'dist_actual_ideal', 'condition'
    ])

for cond in CONDITIONS:
    for ID in SUBJECTS:
        for trial in TRIALS:
            df_trial = mf.make_df_trial(df_all, ID, cond, agents, trial)
            tmp = pd.DataFrame({'completion_time': df_trial['timerTrial'].iloc[-1],
                                'dist_closest': np.mean(df_trial['dist_closest']),
                                'dist_actual_ideal': np.mean(df_trial['dist_actual_ideal']),
                                'condition': cond},
                               index=[f'ID{ID}_{cond}_agents{agents}_trial{trial}'])
            df_sum = pd.concat([df_sum, tmp])

from sklearn.cluster import KMeans
import warnings
warnings.simplefilter('ignore')
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
#Import svm model
from sklearn import svm

X = df_sum[['completion_time', 'dist_closest', 'dist_actual_ideal']]
Y = df_sum['condition']
x_train, y_train = X[:400], Y[:400]
x_test, y_test = X[400:], Y[400:]

#Create a svm Classifier
clf = svm.SVC(kernel='rbf') # Linear Kernel

#Train the model using the training sets
clf.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


kmeans_model = KMeans(n_clusters=3)
clusters = kmeans_model.fit_predict(X)
print('urgent   ', clusters[:232])
print('nonurgent', clusters[232:464])
print('omoiyari ', clusters[464:])



# %% plot functions
mf.plot_traj_per_trials(df_all, ID, cond, agents)
mf.plot_traj_compare_conds(df_all, ID, agents)

mf.plot_dist_compare_conds(df_all, ID, agents, "dist_actual_ideal")
mf.plot_dist_per_cond(df_all, ID, agents, "dist_actual_ideal")

mf.plot_dist_compare_conds(df_all, ID, agents, "dist_closest")
mf.plot_dist_per_cond(df_all, ID, agents, "dist_closest")

# %%
posXtplus1  = pd.concat([df_trial.posX[1:], pd.Series([None])], ignore_index=True)
posYtplus1  = pd.concat([df_trial.posY[1:], pd.Series([None])], ignore_index=True)
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

# %% time series clustering
df_clustering = mf.make_df_for_clustering(df_all, 8, 20, "dist_actual_ideal")
true_labels = ["omoiyari", "isogi", "yukkuri"] * 8

df_clustering = pd.DataFrame()
for ID in tqdm(SUBJECTS):
    for agent in NUM_AGENTS:
        df = mf.make_df_for_clustering(df_all, ID, agent, "dist_actual_ideal")
        df_clustering = pd.concat([df_clustering, df], ignore_index=True)


from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesScalerMeanVariance    
from tslearn.metrics import dtw 

from sklearn.preprocessing import StandardScaler
scaler_std = StandardScaler()
df_clustering = scaler_std.fit_transform(df_clustering)

time_np = to_time_series_dataset(df_clustering.T)

n = 3
km_euclidean = TimeSeriesKMeans(n_clusters=n, metric='dtw')
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

