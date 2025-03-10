# %% Global varibales
SUBJECTS = [subject for subject in range(1, 30)]
CONDITIONS = ['isogi', 'yukkuri', 'omoiyari']
AGENTS = [agent for agent in range(1, 21)]
NUM_AGENTS = [5, 10, 20]
TRIALS = [trial for trial in range(1, 9)]

# %% import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from tslearn.clustering import TimeSeriesKMeans
from typing import Literal
from tqdm import tqdm

# %%
def col(df):
    try:
        for i, c in enumerate(df.columns): print(i, c)
    except AttributeError:
        for i, c in enumerate(df.index): print(i, c)

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
def calc_distance(myX, myY, anotherX, anotherY):
    mypos = np.array([myX, myY])
    anotherpos = np.array([anotherX, anotherY])
    distance = np.linalg.norm(mypos - anotherpos)
    
    return distance

# %% 
def add_cols_dist_others(df):
    df_tmp = pd.DataFrame()
    for i in AGENTS:
        try:
            dist_others = df.apply(lambda df_: calc_distance(
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
    dist_from_start = df.apply(lambda df_: calc_distance(
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
def calc_BrakeRate(df):        
    myinfo = df.loc[:, df.columns.str.startswith('my')]
    brakings = df.loc[:, df.columns.str.startswith('other')]
    brakings = pd.concat([myinfo, brakings], axis=1)    
    
    max_brake = []
    sum_brake = []
    for data in brakings.iterrows():
        brakeRates = []
        for i in range(1, 21):
            brake = BrakingRate(
                data[1]['myMoveX'], data[1]['myMoveY'],
                data[1]['myNextX'], data[1]['myNextY'],
                data[1][f'other{i}MoveX'], data[1][f'other{i}MoveY'],
                data[1][f'other{i}NextX'], data[1][f'other{i}NextY'],
                return_when_undefined=0
            )
            brakeRates.append(brake)
        max_brake.append(max(brakeRates))
        if not np.isnan(sum(brakeRates)): 
            sum_brake.append(sum(brakeRates))
        else:
            sum_brake.append(0)
        
    df['BrakeRate_Max'] = max_brake
    df['BrakeRate_Sum'] = sum_brake
    
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
        2. experiment conditions (isogi, yukkuri, omoiyari)
        3. number of agents (5, 10, 20)
        4. trial (1-8)
                
    ex. df = df_all["ID1"]["omoiyari"]["agents5"]["trial1"]
    """
    pd.options.mode.chained_assignment = None # otherwise there'll be many warnings
    
    df_all = make_empty_hierarchical_df(subjects, CONDITIONS, NUM_AGENTS)
        
    for ID in tqdm(subjects):
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
                    if agent == 20:
                        df_tri = calc_BrakeRate(df_tri)

                    df_all[f"ID{ID}"][cond][f"agents{agent}"][f"trial{trial+1}"] = df_tri
                        
    return df_all

# %% 
def make_df_trial(df_all: pd.DataFrame, 
                  ID: int, 
                  condition: Literal['isogi', 'yukkuri', 'omoiyari'], 
                  num_agents: int, 
                  trial: int) -> pd.DataFrame:
    
    df = df_all[f'ID{ID}'][condition][f"agents{num_agents}"][f"trial{trial}"]
    
    return df
    
# %% 
def make_df_for_clustering(df_all: pd.DataFrame,
                           ID: int, 
                           agents: Literal[5, 10, 20], 
                           column: str
                           ) -> pd.DataFrame:
    
    df_clustering = pd.DataFrame()
    for trial in TRIALS:
        omoiyari = df_all[f"ID{ID}"]["omoiyari"][f"agents{agents}"][f"trial{trial}"][column]
        isogi = df_all[f"ID{ID}"]["isogi"][f"agents{agents}"][f"trial{trial}"][column]
        yukkuri = df_all[f"ID{ID}"]["yukkuri"][f"agents{agents}"][f"trial{trial}"][column]
        
        df_clustering[f"ID{ID}_omoiyari_dist_{trial}"] = omoiyari
        df_clustering[f"ID{ID}_isogi_dist_{trial}"] = isogi
        df_clustering[f"ID{ID}_yukkuri_dist_{trial}"] = yukkuri
        
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

# %%
def plot_traj_per_trials(df_all: pd.DataFrame, 
                         ID: int, 
                         conditions: Literal["isogi", "yukkuri", "omoiyari"], 
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
                      'dist_top12_closest', 'dist_from_start']
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
    plt.tight_layout()
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
    for i in tqdm(range(1, 7)): 
        ts_km = TimeSeriesKMeans(n_clusters=i, metric="dtw", random_state=42) 
        ts_km.fit_predict(df.T) 
        distortions.append(ts_km.inertia_) 
    
    plt.plot(range(1, 7), distortions, marker="o") 
    plt.xticks(range(1, 7)) 
    plt.xlabel("Number of clusters") 
    plt.ylabel("Distortion") 
    plt.show()

# %% 
def plot_result_of_clustering(km_euclidean, time_np, labels_euclidean, n_clusters):
    fig = plt.figure(figsize=(12, 4))
    for i in range(n_clusters):
        ax = fig.add_subplot(1, n_clusters, i+1)
        clus_arr = time_np[labels_euclidean == i]
        for x in clus_arr:
            ax.plot(x.ravel(), 'k-', alpha=0.2)
        ax.plot(km_euclidean.cluster_centers_[i].ravel(), 'r-')
        datanum = np.count_nonzero(labels_euclidean == i)
        ax.text(0.5, max(clus_arr[1])*0.8, f'Cluster{i} : n = {datanum}')
    plt.suptitle('time series clustering')
    plt.show()
    
# %%
def plot_positions(df, time):
    plt.figure(figsize=(8, 8))
    plt.scatter(df.loc[time, 'myNextX'], df.loc[time, 'myNextY'], color='blue')
    plt.scatter(df.loc[time-1, 'myNextX'], df.loc[time-1, 'myNextY'], 
                color='blue', alpha=0.3)
    for i in range(1, 21):
        plt.scatter(df.loc[time, f'other{i}NextX'], df.loc[time, f'other{i}NextY'], 
                    color='gray')
        plt.annotate(i, xy=(df.loc[time, f'other{i}NextX'], 
                            df.loc[time, f'other{i}NextY']))
        plt.scatter(df.loc[time-1, f'other{i}NextX'], 
                    df.loc[time-1, f'other{i}NextY'], 
                    color='gray', alpha=0.3)

# %%
def animate_movements(df_tri, save_as='video.mp4'): 
    fig, ax = plt.subplots()
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
        
    anim = FuncAnimation(fig, update, frames=df_tri.iterrows(), repeat=False, 
                         interval=200, cache_frame_data=False)
    anim.save(save_as)    
    
# %% calculate the judge entropy
def CPx_J(velx1, vely1, posx1, posy1, 
          velx2, vely2, posx2, posy2):
    """
    =IFERROR( 
        (H2*$E2*$B2 - $D2*I2*F2 + $D2*H2*(G2-$C2)) / (H2*$E2 - $D2*I2)
    , "")
    """
    nume = velx2*vely1*posx1 - velx1*vely2*posx2 + velx1*velx2*(posy2 - posy1)
    deno = velx2*vely1 - velx1*vely2
    try:
        val = nume / deno    
        return val
    except ZeroDivisionError:
        return None

def CPy_K(velx1, vely1, posx1, posy1, 
          velx2, vely2, posx2, posy2):
    """
    =IFERROR( 
        (E2/D2) * (J2 - B2) + C2
    , "")
    """
    CPx = CPx_J(velx1, vely1, posx1, posy1, velx2, vely2, posx2, posy2)
    
    try:
        val = (vely1 / velx1) * (CPx - posx1) + posy1
        return val
    except:
        return None

def TTCP0_L(velx1, vely1, posx1, posy1, 
            velx2, vely2, posx2, posy2):
    """
    =IF(
        AND (
            OR ( AND (B2 < J2, D2 > 0), 
                 AND (B2 > J2, D2 < 0)), 
            OR ( AND (C2 < K2, E2 > 0), 
                 AND (C2 > K2, E2 < 0))
            ), 
        SQRT(( J2 - $B2 )^2 + (K2 - $C2)^2) / ( SQRT(($D2^2 + $E2^2 )))
    , "")
    """
    CPx = CPx_J(velx1, vely1, posx1, posy1, velx2, vely2, posx2, posy2)
    CPy = CPy_K(velx1, vely1, posx1, posy1, velx2, vely2, posx2, posy2)
    
    try:
        if ( 
                ( (posx1 < CPx and velx1 > 0) or (posx1 > CPx and velx1 < 0) ) 
            and
                ( (posy1 < CPy and vely1 > 0) or (posy1 > CPy and vely1 < 0) )
            ):
            
            nume = np.sqrt(
                (CPx - posx1)**2 + (CPy - posy1)**2
            )
            deno = np.sqrt(
                (velx1**2 + vely1**2)
            )
            
            val = nume / deno
            return val 
    except TypeError:
        return None
    
    else:
        return None
        
def TTCP1_M(velx1, vely1, posx1, posy1, 
            velx2, vely2, posx2, posy2):
    """
    =IF( 
        AND (
            OR ( AND (F2 < J2, H2 > 0), 
                 AND (F2 > J2, H2 < 0)), 
            OR ( AND (G2 < K2, I2 > 0), 
                 AND (G2 > K2, I2 < 0))
            ), 
        SQRT( (J2 - F2)^2 + (K2 - G2)^2 ) / (SQRT( (H2^2 + I2^2) ))
    , "")

    """
    CPx = CPx_J(velx1, vely1, posx1, posy1, velx2, vely2, posx2, posy2)
    CPy = CPy_K(velx1, vely1, posx1, posy1, velx2, vely2, posx2, posy2)
    
    try:
        if ( 
                ( (posx2 < CPx and velx2 > 0) or (posx2 > CPx and velx2 < 0) ) 
            and
                ( (posy2 < CPy and vely2 > 0) or (posy2 > CPy and vely2 < 0) )
            ):
            
            nume = np.sqrt(
                (CPx - posx2)**2 + (CPy - posy2)**2
            )
            deno = np.sqrt(
                (velx2**2 + vely2**2)
            )
            
            val = nume / deno
            return val
    except TypeError:
        return None
    
    else:
        return None 

def deltaTTCP_N(velx1, vely1, posx1, posy1, 
                velx2, vely2, posx2, posy2,
                return_when_undefined=None):
    """
    =IFERROR( 
        ABS(L2 - M2)
    , -1)
    """
    TTCP0 = TTCP0_L(velx1, vely1, posx1, posy1, velx2, vely2, posx2, posy2)
    TTCP1 = TTCP1_M(velx1, vely1, posx1, posy1, velx2, vely2, posx2, posy2)
    
    try:
        val = abs(TTCP0 - TTCP1)
        return val
    except:
        return return_when_undefined

def Judge_O(velx1, vely1, posx1, posy1, 
            velx2, vely2, posx2, posy2, 
            eta1=-0.303, eta2=0.61):
    """
    =IFERROR(
        1 / (1 + EXP($DB$1[eta1] + $DC$1[eta2]*(M2 - L2)))
    , "")
    """
    TTCP0 = TTCP0_L(velx1, vely1, posx1, posy1, velx2, vely2, posx2, posy2)
    TTCP1 = TTCP1_M(velx1, vely1, posx1, posy1, velx2, vely2, posx2, posy2)
    
    deno = 1 + np.exp(eta1 + eta2*(TTCP1 - TTCP0))
    val = 1 / deno
    return val

def JudgeEntropy(velx1, vely1, posx1, posy1, 
                 velx2, vely2, posx2, posy2):
    """
    =IFERROR( 
        -O2 * LOG(O2) - (1 - O2) * LOG(1 - O2)
    , "")
    """
    Judge = Judge_O(velx1, vely1, posx1, posy1, velx2, vely2, posx2, posy2)
    
    val = -Judge * np.log10(Judge) - (1 - Judge) * np.log10(1 - Judge)
    return val

# %% calculate the braking rate
def equA_Q(velx1, vely1, velx2, vely2):
    """
    = ($D2 - H2)^2 + ($E2 - I2)^2
    """
    val = (velx1 - velx2)**2 + (vely1 - vely2)**2
    return val
    
def equB_R(velx1, vely1, posx1, posy1, 
           velx2, vely2, posx2, posy2):
    """
    = (2*($D2 - H2)*($B2 - F2)) + (2*($E2 - I2)*($C2 - G2))
    """
    val = (2 * (velx1 - velx2) * (posx1 - posx2)) + \
          (2 * (vely1 - vely2) * (posy1 - posy2))
    return val

def equC_S(posx1, posy1, posx2, posy2):
    """
    = ($B2 - F2)^2 + ($C2 - G2)^2
    """
    val = (posx1 - posx2)**2 + (posy1 - posy2)**2
    return val

def TCPA_T(equA, equB): 
    """
    = -(R2 / (2*Q2))
    """
    val = -(equB / (2*equA))
    return val

def DCPA_U(equA, equB, equC):
    """
    = SQRT( (-(R2^2) + (4*Q2*S2)) / (4*Q2) ) 
    """
    val = np.sqrt(
        (-(equB**2) + (4*equA*equC)) / (4*equA)
    )
    return val

def BrakingRate(velx1, vely1, posx1, posy1, 
                velx2, vely2, posx2, posy2, 
                return_when_undefined=None,
                a1=-0.034, b1=3.348, c1=4.252, d1=-0.003):
    """
    a1: -5.145 (-0.034298)
    b1: 3.348 (3.348394)
    c1: 4.286 (4.252840)
    d1: -13.689 (-0.003423)
    
    =IF(T2 < 0, "", 
    IFERROR(
        (1 / (1 + EXP(-($DD$1[c1] + ($DE$1[d1]*T2*1000))))) * 
        (1 / (1 + EXP(-($DF$1[b1] + ($DG$1[a1]*30*U2)))))
        , "")
    )
    """
    equA = equA_Q(velx1, vely1, velx2, vely2)
    equB = equB_R(velx1, vely1, posx1, posy1, velx2, vely2, posx2, posy2)
    equC = equC_S(posx1, posy1, posx2, posy2)
    TCPA = TCPA_T(equA, equB)
    DCPA = DCPA_U(equA, equB, equC)
    
    if TCPA < 0:
        return return_when_undefined
    else:
        term1 = (1 / (1 + np.exp(-(c1 + (d1*TCPA)))))
        term2 = (1 / (1 + np.exp(-(b1 + (a1*DCPA)))))
        val = term1 * term2
        return val
    
# %% awareness model
def calculate_angle(line1, line2):
    """
    2直線の角度を求める関数
    line1, line2: 各直線を表す2点 [(x1, y1), (x2, y2)]
    """
    # 傾きを計算
    def slope(point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        if x2 - x1 == 0:  # 垂直な直線の場合
            return np.inf  # 無限大を返す
        return (y2 - y1) / (x2 - x1)
    
    m1 = slope(*line1)
    m2 = slope(*line2)
    
    # 垂直チェック
    if m1 == np.inf and m2 == np.inf:
        return 0  # 同じ垂直方向
    elif m1 == np.inf or m2 == np.inf:
        return 90  # 片方が垂直の場合
    
    # 角度を計算
    angle_rad = np.arctan(np.abs((m2 - m1) / (1 + m1 * m2)))
    return angle_rad


def calc_Nic(df, agent):
    my_pos = (df['B_posx1'], df['C_posy1'])
    other_pos = (df['F_posx2'], df['G_posy2'])
    cp = ( (my_pos[0] + other_pos[0]) / 2, (my_pos[1] + other_pos[1]) / 2 )
    dist_cp_me = calc_distance(cp[0], cp[1], my_pos[0], my_pos[1])
    
    Nic_agents = []
    for i in range(1, 21):
        other_pos = (df[f'other{i}NextX'], df[f'other{i}NextY'])
        dist_cp_other = calc_distance(cp[0], cp[1], other_pos[0], other_pos[1])
        if dist_cp_other <= dist_cp_me and not i == agent:
            Nic_agents.append(i)
    
    return Nic_agents


def awareness_model(deltaTTCP, Px, Py, Vself, Vother, theta, Nic):
    """
    Inputs
        deltaTTCP: 自分のTTCPから相手のTTCPを引いた値 (second)
        Px: 自分から見た相手の相対位置 (x座標m)
        Py: 自分から見た相手の相対位置 (y座標m)
        Vself: 自分の歩行速度 (m/s)
        Vother: 相手の歩行速度 (m/s)
        theta: 自分の向いている方向と相手の位置の角度差 (rad)
        Nic: 円内他歩行者数 (人)
    Output
        0(注視しない) - 1 (注視する)
    """
    if deltaTTCP == None:
        return 0
    
    deno = 1 + np.exp(
        -(-1.2 +0.018*deltaTTCP -0.1*Px -1.1*Py \
          -0.25*Vself +0.29*Vother -2.5*theta -0.62*Nic)    
    )
    val = 1 / deno
    return val

# %% funcs for crossing_experiment.py
import funcs_calc_index as ci

def make_column_names_shorter(df):
    df = df[['/vrpn_client_node/body_0/pose/field.pose.position.x',
             '/vrpn_client_node/body_0/pose/field.pose.position.z',
             'body_0_vel_x', 
             'body_0_vel_z',
             '/vrpn_client_node/body_1/pose/field.pose.position.x',
             '/vrpn_client_node/body_1/pose/field.pose.position.z',
             'body_1_vel_x', 
             'body_1_vel_z']]
    
    old_new_cols = {
        '/vrpn_client_node/body_0/pose/field.pose.position.x': 'B_posx1',
        '/vrpn_client_node/body_0/pose/field.pose.position.z': 'C_posy1',
        'body_0_vel_x': 'D_velx1',
        'body_0_vel_z': 'E_vely1',
        '/vrpn_client_node/body_1/pose/field.pose.position.x': 'F_posx2',
        '/vrpn_client_node/body_1/pose/field.pose.position.z': 'G_posy2',
        'body_1_vel_x': 'H_velx2',
        'body_1_vel_z': 'I_vely2'
    }
    
    df = df.rename(columns=old_new_cols)
    
    return df

def calculate_indexes(df):
    df['J_CPx'] = ci.J_CPx(df)
    df['K_CPy'] = ci.K_CPy(df)
    df['L_TTCP0'] = df.apply(ci.L_TTCP0, axis=1)
    df['M_TTCP1'] = df.apply(ci.M_TTCP1, axis=1)
    df['N_deltaTTCP'] = ci.N_deltaTTCP(df)
    df['O_Judge'] = ci.O_Judge(df)
    df['P_JudgeEntropy'] = ci.P_JudgeEntropy(df) 
    df['Q_equA'] = ci.Q_equA(df)
    df['R_equB'] = ci.R_equB(df)
    df['S_equC'] = ci.S_equC(df)
    df['T_TCPA'] = ci.T_TCPA(df) 
    df['U_DCPA'] = ci.U_DCPA(df)
    df['V_BrakingRate'] = df.apply(ci.V_BrakingRate, axis=1)
    df['W_distance'] = df.apply(ci.W_distance, axis=1)
    
    return df

# animte the moving behaviors and the transactions of indexes
def animate_traj(df, save_as='crossing.mp4'):
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'BIZ UDGothic'
    
    fig, ax = plt.subplots(2, 3, figsize=(14, 10))
    
    headwidth = 8
    headlength = 8
    linewidth = 3
    BrakingRate = []
    distance = []
    JudgeEntropy = []
    TCPA = []
    DCPA = []
    deltaTTCP = []
    TCPA_ymax = np.max(df['T_TCPA'])
    DCPA_ymax = np.max(df['U_DCPA'])
    deltaTTCP_ymax = np.max(df['N_deltaTTCP'])
    xmax = len(df)
    tminus1 = None
    
    df.fillna(-0.1, inplace=True)
    
    def update(frame):
        global tminus1
        if np.isnan(frame[1]['P_JudgeEntropy']) == False:
            JudgeEntropy.append(frame[1]['P_JudgeEntropy'])
        else:
            JudgeEntropy.append(0)
        if np.isnan(frame[1]['V_BrakingRate']) == False:
            BrakingRate.append(frame[1]['V_BrakingRate'])
        else:
            BrakingRate.append(0)
        distance.append(frame[1]['W_distance'])
        TCPA.append(frame[1]['T_TCPA'])
        DCPA.append(frame[1]['U_DCPA'])
        deltaTTCP.append(frame[1]['N_deltaTTCP'])
        
        # ax[0,0] 歩行者の動き
        ax[0,0].cla()
        ax[0,0].set_title('歩行者の動き')
        ax[0,0].set_xlim(-5, 5)
        ax[0,0].set_ylim(-5, 5)
        ax[0,0].grid()
        ax[0,0].scatter(frame[1]['B_posx1'], frame[1]['C_posy1'], s=80, c='blue', alpha=0.8)
        ax[0,0].scatter(frame[1]['F_posx2'], frame[1]['G_posy2'], s=80, c='red', alpha=0.8)
        ax[0,0].scatter(df['B_posx1'], df['C_posy1'], c='blue', alpha=0.03)
        ax[0,0].scatter(df['F_posx2'], df['G_posy2'], c='red', alpha=0.03)
        if not frame[0] == 0:
            ax[0,0].scatter(tminus1[1]['B_posx1'], tminus1[1]['C_posy1'], s=80, c='blue', alpha=0.5)
            ax[0,0].scatter(tminus1[1]['F_posx2'], tminus1[1]['G_posy2'], s=80, c='red', alpha=0.5)
        
        if not frame[0]+1 == len(df):
            ax[0,0].annotate('', xy=(df['B_posx1'][frame[0]+1], df['C_posy1'][frame[0]+1]), 
                             xytext=(frame[1]['B_posx1'], frame[1]['C_posy1']),
                             arrowprops=dict(shrink=0, width=1, headwidth=headwidth, 
                                             headlength=headlength,
                                             facecolor='skyblue', edgecolor='blue'))
            
            ax[0,0].annotate('', xy=(df['F_posx2'][frame[0]+1], df['G_posy2'][frame[0]+1]), 
                             xytext=(frame[1]['F_posx2'], frame[1]['G_posy2']),
                             arrowprops=dict(shrink=0, width=1, headwidth=headwidth, 
                                             headlength=headlength,
                                             facecolor='pink', edgecolor='red'))
        tminus1 = frame
        
        # ax[0,1] deltaTTCP (2者の交点までの時間差)
        ax[0,1].cla()
        ax[0,1].set_title('ΔTTCP (2者の交点までの時間差)')
        ax[0,1].set_xlim(0, xmax+1)
        ax[0,1].set_ylim(0, deltaTTCP_ymax+0.5)
        ax[0,1].grid()
        ax[0,1].plot(deltaTTCP, lw=linewidth, color='green', alpha=0.7)
        ax[0,1].plot(df['N_deltaTTCP'], lw=linewidth, color='green', alpha=0.1)
        
        # ax[0,2] 判断エントロピー 
        ax[0,2].cla()
        ax[0,2].set_title('判断エントロピー：0(確定的)～1(曖昧)', size=13)
        ax[0,2].set_xlim(0, xmax+1)
        ax[0,2].set_ylim(-0.01, 1)
        ax[0,2].grid()
        ax[0,2].plot(JudgeEntropy, lw=linewidth, color='crimson', alpha=0.7)
        ax[0,2].plot(df['P_JudgeEntropy'], lw=linewidth, color='crimson', alpha=0.08)
        
        # ax[1,0] TCPA (最接近点までの時間)
        ax[1,0].cla()
        ax[1,0].set_title('TCPA (最接近点までの時間)')
        ax[1,0].set_xlim(0, xmax+1)
        ax[1,0].set_ylim(0, TCPA_ymax+0.5)
        ax[1,0].grid()
        ax[1,0].plot(TCPA, lw=linewidth, color='gray')
        ax[1,0].plot(df['T_TCPA'], lw=linewidth, color='gray', alpha=0.12)
        ax[1,0].text(x=0.3, y=-0.2, s=f'時間：{frame[0]} (100ms)', 
                     size=13, transform=ax[1,1].transAxes)    
        
        # ax[1,1] DCPA (最接近距離)
        ax[1,1].cla()
        ax[1,1].set_title('DCPA (最接近距離)')
        ax[1,1].set_xlim(0, xmax+1)
        ax[1,1].set_ylim(0, DCPA_ymax+0.5)
        ax[1,1].grid()
        ax[1,1].plot(DCPA, lw=linewidth, color='chocolate', alpha=0.7)
        ax[1,1].plot(df['U_DCPA'], lw=linewidth, color='chocolate', alpha=0.12)
        
        # ax[1,2] ブレーキ率
        ax[1,2].cla()
        ax[1,2].set_title('ブレーキ率：0(踏まない)～1(踏む)')
        ax[1,2].set_xlim(0, xmax+1)
        ax[1,2].set_ylim(-0.01, 1)
        ax[1,2].grid()
        ax[1,2].plot(BrakingRate, lw=linewidth, color='purple', alpha=0.7)
        ax[1,2].plot(df['V_BrakingRate'], lw=linewidth, color='purple', alpha=0.1)
        
    anim = FuncAnimation(fig, update, frames=df.iterrows(), repeat=False, 
                         interval=250, cache_frame_data=False)
    # plt.show()
    anim.save(save_as, writer='ffmpeg')
    plt.close()
