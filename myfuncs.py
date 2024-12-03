# %% Global varibales
SUBJECTS = [subject for subject in range(1, 30)]
CONDITIONS = ['urgent', 'nonurgent', 'omoiyari']
AGENTS = [agent for agent in range(1, 21)]
NUM_AGENTS = [5, 10, 20]
TRIALS = [trial for trial in range(1, 9)]

# %% import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
sns.set_theme()
from tslearn.clustering import TimeSeriesKMeans

from tqdm import tqdm
from typing import Literal
    
# %% 
def make_start_from_UL(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make all trials' start points (30, 30) and add new columns reperesenting 
    those new X and Y positions (posX, PosY).
    """
    if "UR" in df["type"].values[0]: # (940, 30)
        df["posX"] = df["myNextX"].map(lambda x: abs(x - 910))
        df["posY"] = df["myNextY"]
        
    if "LR" in df["type"].values[0]: # (940, 940)
        df["posX"] = df["myNextX"].map(lambda x: abs(x - 910))
        df["posY"] = df["myNextY"].map(lambda y: abs(y - 910))
        
    if "LL" in df["type"].values[0]: # (30, 940)
        df["posX"] = df["myNextX"]
        df["posY"] = df["myNextY"].map(lambda y: abs(y - 910))

    if "UL" in df["type"].values[0]: # (30, 30)
        df["posX"] = df["myNextX"]
        df["posY"] = df["myNextY"]

    return df

# %% 
def drop_unmoved_from_start(df):
    for i, (x, y) in enumerate(zip(df["posX"], df["posY"])):
        if x != 30 or y != 30:
            df_dropped = df[i-1:]
            df_dropped.reset_index(inplace=True)
            break
    
    return df_dropped

# %% 
def _calc_ideal_positions(x1: int, 
                          y1: int, 
                          goalx: int = 880, 
                          goaly: int = 880) -> tuple[int, int]:
    """
    Return the ideal next XY positions by calculating the intercation points between
    the straight line to the goal from the current position and the maximum movable 
    areas.
    """
    xmin, xmax = x1-20, x1+20
    ymin, ymax = y1-20, y1+20
    
    if x1 == goalx:
        return x1, ymax
    if y1 == goaly:
        return xmax, y1
            
    slope = (goalx - x1) / (goaly - y1)
    y = slope * (xmax - x1) + y1
    x = (ymax - y1) / slope + x1
    y, x = np.round(y, 3), np.round(x, 3)

    if y <= ymax and y >= ymin:
        return xmax, y
    else: 
        return x, ymax
    
# %% 
def add_cols_ideal_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'idealNextX' and 'idealNextY' columns to the dataframe.
    """
    posIdeal = df.apply(
        lambda df: _calc_ideal_positions(df["posX"], df["posY"]), 
        axis=1)
    idealX, idealY = [], []
    for i in range(len(posIdeal)):
        idealX.append(posIdeal[i][0])
        idealY.append(posIdeal[i][1])
    df['idealNextX'] = idealX
    df['idealNextY'] = idealY
    
    return df

# %% 
def calc_dist_actual_ideal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the distance between actual xy positions and the ideal xy positions.
    Finally added the column named "dist_actual_ideal" to the dataframe.
    """
    dists = [None]
    for rx, ry, ix, iy in zip(df["posX"][1:], df["posY"][1:],
                              df["idealNextX"][:-1], df["idealNextX"][:-1]):
        actualpos = np.array([rx, ry])
        idealpos = np.array([ix, iy])
        distance = np.linalg.norm(actualpos - idealpos)
        dists.append(distance)
        
    df["dist_actual_ideal"] = dists
    
    return df

# %% 
def drop_unnecessary_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns whose values are all None or 0.
    """
    df.drop("Unnamed: 117", axis=1, inplace=True)
    cols_to_drop = []
    for i in [6, 11]:
        if all(x == 0 for x in df[f"other{i}NextX"]):
            for j in range(i, len(AGENTS)+1):
                other_cols = [f"other{j}NextX", f"other{j}NextY", 
                              f"other{j}MoveX", f"other{j}MoveY",
                              f"other{j}Coll"]    
                cols_to_drop.extend(other_cols)
            
    df.drop(cols_to_drop, axis=1, inplace=True)        
    
    return df

# %% 
def _calc_distance(myX, myY, anotherX, anotherY):
    mypos = np.array([myX, myY])
    anotherpos = np.array([anotherX, anotherY])
    distance = np.linalg.norm(mypos - anotherpos)
    
    return distance

# %% 
def add_cols_dist_others(df):
    df_tmp = pd.DataFrame()
    for i in AGENTS:
        try:
            dist_others = df.apply(lambda df_: _calc_distance(
               df_["myNextX"], df_["myNextY"], 
               df_[f"other{i}NextX"], df_[f"other{i}NextY"]
               ), axis=1)
            df_tmp[f"distOther{i}"] = dist_others
        except KeyError:
            break
    
    newdf = pd.concat([df, df_tmp], axis=1)
    
    return newdf

# %% 
def add_col_dist_from_start(df):
    dist_from_start = df.apply(lambda df_: _calc_distance(
        df_["posX"], df_["posY"], 30, 30
        ), axis=1)
    df["dist_from_start"] = dist_from_start
    
    return df

# %% 
def calc_closest_others(df):
    df_others = df.filter(like="distOther")
    dist_closest = df_others.apply(min, axis=1)
    df['dist_closest'] = dist_closest
    
    return df

# %% 
def _dist_sum_1st2nd_closest(series):
    array_sorted = series.sort_values(ascending=True)
    sum_top12_closest = sum(array_sorted[0:2])
    
    return sum_top12_closest

# %% 
def add_col_dist_top12_closest(df):
    df_others = df.filter(like="distOther")
    dist_1st2nd_closest = df_others.apply(
        lambda series: _dist_sum_1st2nd_closest(series), 
        axis=1)
    df['dist_top12_closest'] = dist_1st2nd_closest
    
    return df

# %% 
def preprocess(df):
    df.reset_index(drop=True, inplace=True)
    df = make_start_from_UL(df)
    df = drop_unmoved_from_start(df)
    df = add_cols_ideal_positions(df)
    df = calc_dist_actual_ideal(df)
    df = drop_unnecessary_cols(df)
    df = add_cols_dist_others(df)
    df = calc_closest_others(df)
    df = add_col_dist_top12_closest(df)
    df = add_col_dist_from_start(df)
    
    return df

# %% 
def make_empty_hierarchical_df(SUBJECTS, CONDITIONS, NUM_AGENTS):
    df_empty = {}
    
    for subjectID in SUBJECTS:
        df_empty[f"ID{subjectID}"] = {}
        
    for subjectID in SUBJECTS:
        for condition in CONDITIONS:
            df_empty[f"ID{subjectID}"][condition] = {}
    
    for subjectID in SUBJECTS:
        for condition in CONDITIONS:
            for agent in NUM_AGENTS:
                df_empty[f"ID{subjectID}"][condition][f"agents{agent}"] = {}

    return df_empty

# %% 
def make_dict_of_all_info(subjects: list[int] = SUBJECTS,
                          folder_path: str = "04_RawData") -> pd.DataFrame:
    """
    Make a dictionary that contains all subjects' records.
    The dictionary's hierarchy is: 
        1. participant's ID
        2. experiment conditions (urgent, nonurgent, omoiyari)
        3. number of agents (5, 10, 20)
        4. trial (1-8)
                
    ex. df = df_all["ID1"]["omoiyari"]["agents5"]["trial1"]
    """
    pd.options.mode.chained_assignment = None # otherwise there'll be many warnings
    
    df_all = make_empty_hierarchical_df(SUBJECTS, CONDITIONS, NUM_AGENTS)
        
    for ID in tqdm(SUBJECTS):
        for cond in CONDITIONS:
        
            df = pd.read_csv(f'{folder_path}/{ID}_{cond}.csv')
            
            df_5 = df.query('type.str.contains("05")')
            df_10 = df.query('type.str.contains("10")')
            df_20 = df.query('type.str.contains("20")')
            
            dfs = [df_5, df_10, df_20]
            
            trialnum5 = list(set(df_5['trial']))
            trialnum10 = list(set(df_10['trial']))
            trialnum20 = list(set(df_20['trial']))
            
            trialnums = [trialnum5, trialnum10, trialnum20]
            
            for df_agent, trialnum, agent in zip(dfs, trialnums, NUM_AGENTS):
                for trial in range(len(trialnum5)):
                    
                    df_tri = df_agent.query('trial == @trialnum[@trial]')
                    df_tri = preprocess(df_tri)

                    df_all[f"ID{ID}"][cond][f"agents{agent}"][f"trial{trial+1}"] = df_tri
                        
    return df_all

# %% 16
def make_df_trial(df_all: pd.DataFrame, 
                  ID: int, 
                  condition: Literal['urgent', 'nonurgent', 'omoiyari'], 
                  num_agents: int, 
                  trial: int) -> pd.DataFrame:
    
    df = df_all[f'ID{ID}'][condition][f"agents{num_agents}"][f"trial{trial}"]
    
    return df
    
# %% 17
def make_df_for_clustering(df_all: pd.DataFrame,
                           ID: int, 
                           agents: Literal[5, 10, 20], 
                           column: str
                           ) -> pd.DataFrame:
    
    df_clustering = pd.DataFrame()
    for trial in TRIALS:
        omoiyari = df_all[f"ID{ID}"]["omoiyari"][f"agents{agents}"][f"trial{trial}"][column]
        urgent = df_all[f"ID{ID}"]["urgent"][f"agents{agents}"][f"trial{trial}"][column]
        nonurgent = df_all[f"ID{ID}"]["nonurgent"][f"agents{agents}"][f"trial{trial}"][column]
        
        df_clustering[f"ID{ID}_omoiyari_dist_{trial}"] = omoiyari
        df_clustering[f"ID{ID}_urgent_dist_{trial}"] = urgent
        df_clustering[f"ID{ID}_nonurgent_dist_{trial}"] = nonurgent
        
        df_clustering.dropna(inplace=True)
    
    return df_clustering

# %%
def find_max_length(df_all, return_as_list=False):
    lengths = []
    for ID in SUBJECTS:
        for cond in CONDITIONS:
            for trial in TRIALS:
                tmp = make_df_trial(df_all, ID, cond, 20, trial)
                lengths.append(len(tmp))
    max_length = max(lengths)
    
    if return_as_list == True:
        return lengths
    else:
        return max_length

# %%
def pad_with_nan(df, max_length):
    df_nan = pd.DataFrame(index=range(len(df), max_length), columns=df.columns)    
    df_with_nan = pd.concat([df, df_nan], axis=0)    

    return df_with_nan

# %%
def make_arr_for_train_test(df_all, features, conds, ylabs):
    max_length = find_max_length(df_all, return_as_list=False)  
    
    arr = np.zeros((232*len(conds), max_length, len(features)))
    idx = 0
    
    for ID in tqdm(SUBJECTS):
        df_tri = pd.DataFrame()
        for trial in TRIALS:
            for cond in conds:
                df_tmp = make_df_trial(df_all, ID, cond, 20, trial)
                df_tri = pd.concat([df_tri, df_tmp], axis=1)
            
        df_tri = df_tri[features]
        
        # standardize the values within the individual (0-1)
        for feature in features:
            df_tri[feature] /= np.max(df_tri[feature])  
        
        df_tri = pad_with_nan(df_tri, max_length)    
            
        data_per_person = len(TRIALS) * len(conds)
        
        for i in range(data_per_person):
            cols_per_tri = [j for j in range(i, df_tri.shape[1], data_per_person)]
            tmp_arr = df_tri.iloc[:, cols_per_tri]
            arr[idx] = tmp_arr
            idx += 1
        
    return arr

# %% def make_arr_for_train_test(df_all, features, conds, ylabs):
    max_length = find_max_length(df_all, return_as_list=False)  
    
    arr = np.zeros((232*len(conds), max_length, len(features)))
    idx = 0
    
    for ID in tqdm(SUBJECTS):
        df_tri = pd.DataFrame()
        for trial in TRIALS:
            for cond in conds:
                df_tmp = make_df_trial(df_all, ID, cond, 20, trial)
                df_tri = pd.concat([df_tri, df_tmp], axis=1)
            
        df_tri = df_tri[features]
        
        # standardize the values within the individual (0-1)
        for feature in features:
            df_tri[feature] /= np.max(df_tri[feature])  
        
        df_tri = pad_with_nan(df_tri, max_length)    
            
        data_per_person = len(TRIALS) * len(conds)
        
        for i in range(data_per_person):
            cols_per_tri = [j for j in range(i, df_tri.shape[1], data_per_person)]
            tmp_arr = df_tri.iloc[:, cols_per_tri]
            arr[idx] = tmp_arr
            idx += 1
        
    return arr

# %%
def plot_traj_per_trials(df_all: pd.DataFrame, 
                         ID: int, 
                         conditions: Literal["urgent", "nonurgent", "omoiyari"], 
                         num_agents: Literal[5, 10, 20]) -> None:
    """
    Plot each trial's trajectory in different axes(2x4).
    """
    fig = plt.figure(figsize=(10, 6), tight_layout=True)
    for trial, color in zip(TRIALS, mcolors.TABLEAU_COLORS):
        df = df_all[f"ID{ID}"][conditions][f"agents{num_agents}"][f"trial{trial}"]
        ax = fig.add_subplot(2, 4, trial)
        for x, y in zip(df['posX'], df['posY']):
            ax.scatter(x, y, color=color, alpha=.5)
        ax.set_title(f"trial{trial}")
    plt.suptitle(f"ID{ID}_{conditions}_agents{num_agents}")
    plt.show()
    
# %% 
def plot_traj_compare_conds(df_all: pd.DataFrame, 
                            ID: int, 
                            num_agents: Literal[5, 10, 20]) -> None:
    """
    Plot each trial's trajectory for all experiment conditions.
    Figure has 1x3 axes.
    """
    fig = plt.figure(figsize=(10, 4), tight_layout=True)
    with tqdm(total=len(CONDITIONS)*len(TRIALS)) as pbar:
        for i, cond in enumerate(CONDITIONS):
            ax = fig.add_subplot(1, 3, i+1)
            for trial, color in zip(TRIALS, mcolors.TABLEAU_COLORS):
                df = df_all[f"ID{ID}"][cond][f"agents{num_agents}"][f"trial{trial}"]
                for x, y in zip(df['posX'], df['posY']):
                    ax.scatter(x, y, color=color, alpha=.5)
                pbar.update(1)
            ax.set_title(f"ID{ID}_{cond}_agents{num_agents}")
    print("plotting...")
    plt.show()

# %% 
def plot_dist_compare_conds(
        df_all: pd.DataFrame,
        ID: int, 
        agents: Literal[5, 10, 20], 
        dist: Literal['dist_actual_ideal', 'dist_closest', 
                      'dist_top12_closest', 'dist_from_start']
        ) -> None:
    """
    Plot information of distance in one axis. Figures of each condition are overlapped.
    """
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

# %% 
def plot_dist_per_cond(
        df_all: pd.DataFrame,
        ID: int, 
        agents: Literal[5, 10, 20], 
        dist: Literal['dist_actual_ideal', 'dist_closest', 
                      'dist_top12_closest', 'dist_from_start'],
        nihongo = False
        ) -> None:
    """
    Plot information of distance in separated axes (1, 3). 
    """ 
    fig, ax = plt.subplots(1, 3, figsize=(10, 4), sharex="all", sharey="all", tight_layout=True)
    
    for i, (cond, color) in enumerate(zip(CONDITIONS, mcolors.TABLEAU_COLORS)):
        df_small = df_all[f'ID{ID}'][cond][f'agents{agents}']
        for tri in TRIALS:
            if tri == TRIALS[0]:
                ax[i].plot(df_small[f'trial{tri}'][dist], color=color, alpha=.7, label=cond)
            else:
                ax[i].plot(df_small[f'trial{tri}'][dist], color=color, alpha=.7)
        ax[i].legend(loc="upper right")
    plt.suptitle(f"{dist} ID{ID} agents{agents}")
    plt.show()

# %% 
def plot_all_dist_compare_conds(
        df_all: pd.DataFrame,
        subjects: list[int], 
        agents: Literal[5, 10, 20], 
        dist: Literal['dist_actual_ideal', 'dist_closest', 
                      'dist_top12_closest', 'dist_from_start']
        ) -> None:
    """
    Plot information of distance in one axis. Figures of each condition are overlapped.
    """
    fig = plt.figure(figsize=(10, 6), tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    for id_ in tqdm(SUBJECTS):
        for cond, color in zip(CONDITIONS, mcolors.TABLEAU_COLORS):
            df_small = df_all[f'ID{id_}'][cond][f'agents{agents}']
            for tri in TRIALS:
                if tri == TRIALS[0] and id_ == 1:
                    ax.plot(df_small[f'trial{tri}'][dist], color=color, alpha=.2, label=cond)
                else:
                    ax.plot(df_small[f'trial{tri}'][dist], color=color, alpha=.2)
    ax.set_title(f"{dist} agents{agents}")
    plt.legend()
    plt.show()

# %% 
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

# %% 
def plot_result_of_clustering(km_euclidean, time_np, labels_euclidean, n_clusters):
    fig = plt.figure(figsize=(12, 4))
    for i in range(n_clusters):
        ax = fig.add_subplot(1, 3, i+1)
        clus_arr = time_np[labels_euclidean == i]
        for x in clus_arr:
            ax.plot(x.ravel(), 'k-', alpha=0.2)
        ax.plot(km_euclidean.cluster_centers_[i].ravel(), 'r-')
        datanum = np.count_nonzero(labels_euclidean == i)
        ax.text(0.5, max(clus_arr[1])*0.8, f'Cluster{i} : n = {datanum}')
    plt.suptitle('time series clustering')
    plt.show()
    
    
