import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
from typing import Literal

"""
1. show_all
2. make_start_from_UL
3. _calc_ideal_positions
4. add_cols_ideal_positions
5. calc_dist_real_ideal
6. drop_unnecessary_cols
7. _calc_distance
8. add_cols_dist_others
9. calc_closest_others
10. preprocess
11. make_dict_containing_all_info
12. plot_traj_per_trials
13. plot_traj_compare_conds
"""

# %%
def show_all(obj):
    for i in obj: print(i)
    
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
    posIdeal = df.apply(lambda df: _calc_ideal_positions(df["posX"], df["posY"]), axis=1)
    idealX, idealY = [], []
    for i in range(len(posIdeal)):
        idealX.append(posIdeal[i][0])
        idealY.append(posIdeal[i][1])
    df['idealNextX'] = idealX
    df['idealNextY'] = idealY
    
    return df

# %%
def calc_dist_real_ideal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the distance between real xy positions and the ideal xy positions.
    Finally added the column named "dist_real_ideal" to the dataframe.
    """
    dists = [None]
    for rx, ry, ix, iy in zip(df["posX"][1:], df["posY"][1:],
                              df["idealNextX"][:-1], df["idealNextX"][:-1]):
        realpos = np.array([rx, ry])
        idealpos = np.array([ix, iy])
        distance = np.linalg.norm(realpos - idealpos)
        dists.append(distance)
        
    df["dist_real_ideal"] = dists
    
    return df

# %%
def drop_unnecessary_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns whose values are all None or 0.
    """
    df.drop("Unnamed: 117",axis=1, inplace=True)
    cols_to_drop = []
    for i in range(6, 12, 5):
        if all(x == 0 for x in df[f"other{i}NextX"]):
            for j in range(i, 21):
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
    for i in range(1, 21):
        try:
            dist_others = df.apply(lambda df_: _calc_distance(
               df_["myNextX"], df_["myNextY"], df_[f"other{i}NextX"], df_[f"other{i}NextY"]
               ), axis=1)
            df_tmp[f"distOther{i}"] = dist_others
        except KeyError:
            break
    
    newdf = pd.concat([df, df_tmp], axis=1)
    
    return newdf

# %%
def calc_closest_others(df):
    df_others = df.filter(like="distOther")
    closest_dists = df_others.apply(min, axis=1)
    df['closest_dists'] = closest_dists
    
    return df

# %%
def preprocess(df):
    df.reset_index(drop=True, inplace=True)
    df = make_start_from_UL(df)
    df = add_cols_ideal_positions(df)
    df = calc_dist_real_ideal(df)
    df = drop_unnecessary_cols(df)
    df = add_cols_dist_others(df)
    df = calc_closest_others(df)
    
    return df

# %%
def make_dict_containing_all_info(num_subjects: int,
                                  folder_path: str = "04_RawData") -> pd.DataFrame:
    """
    Make a dictionary that contains all subjects' records.
    The dictionary's hierarchy is: 
        participant's ID:
            experiment conditions (control, urgent, nonurgent, omoiyari):
                trial (trial number and number of agents)
                
    ex. df_1_omoiyari_5_1 = df_all_subjects["ID1"]["omoiyari"]["agents5_tri1"]
    """
    pd.options.mode.chained_assignment = None # otherwise there'll be many warnings
    
    exp_conditions = ["urgent", "nonurgent", "omoiyari"] #, "control"]
    
    df_all_subjects = {}
    for subjectID in range(1, num_subjects+1):
        df_all_subjects[f"ID{subjectID}"] = {}
    
    with tqdm(total=len(exp_conditions)*num_subjects) as pbar:
        for condition in exp_conditions:
            for ID in range(1, 30):
            
                df = pd.read_csv(f'{folder_path}/{ID}_{condition}.csv')
                
                df_5 = df.query('type.str.contains("05")')
                df_10 = df.query('type.str.contains("10")')
                df_20 = df.query('type.str.contains("20")')
                
                trialnum5 = list(set(df_5['trial']))
                trialnum10 = list(set(df_10['trial']))
                trialnum20 = list(set(df_20['trial']))
                assert len(trialnum5) == len(trialnum10) == len(trialnum20), "all trials must have same lengths"
                
                dfs_per_trials = {}
                for trial in range(len(trialnum5)):
                    
                    df_5_tri = df_5.query("trial == @trialnum5[@trial]")
                    df_5_tri = preprocess(df_5_tri)
                    
                    df_10_tri = df_10.query("trial == @trialnum10[@trial]")
                    df_10_tri = preprocess(df_10_tri)
                    
                    df_20_tri = df_20.query("trial == @trialnum20[@trial]")
                    df_20_tri = preprocess(df_20_tri)
                    
                    dfs_per_trials[f"agents5_tri{trial+1}"] = df_5_tri
                    dfs_per_trials[f"agents10_tri{trial+1}"] = df_10_tri
                    dfs_per_trials[f"agents20_tri{trial+1}"] = df_20_tri
                                    
                df_all_subjects[f"ID{ID}"][f"{condition}"] = dfs_per_trials
                pbar.update(1)

    return df_all_subjects

# %%
def plot_traj_per_trials(df_all_subjects: pd.DataFrame, 
                         ID: int, 
                         conditions: Literal["urgent", "nonurgent", "omoiyari"], 
                         num_agents: Literal[5, 10, 20]) -> None:
    """
    Plot each trial's trajectory in different axes(2x4).
    """
    fig = plt.figure(figsize=(12, 6))
    for trial, color in zip(range(1, 9), mcolors.TABLEAU_COLORS):
        df = df_all_subjects[f"ID{ID}"][conditions][f"agents{num_agents}_tri{trial}"]
        ax = fig.add_subplot(2, 4, trial)
        for x, y in zip(df['posX'], df['posY']):
            ax.scatter(x, y, color=color, alpha=.5)
        ax.set_title(f"tri{trial}")
    plt.suptitle(f"ID{ID}_{conditions}_agents{num_agents}")
    plt.show()
    
# %%
def plot_traj_compare_conds(df_all_subjects: pd.DataFrame, 
                            ID: int, 
                            num_agents: Literal[5, 10, 20]) -> None:
    """
    Plot each trial's trajectory for all experiment conditions.
    Figure has 1x3 axes.
    """
    fig = plt.figure(figsize=(12, 6))
    conditions = ["urgent", "nonurgent", "omoiyari"]
    num_all_trials = 8
    with tqdm(total=len(conditions)*num_all_trials) as pbar:
        for i, condition in enumerate(conditions):
            ax = fig.add_subplot(1, 3, i+1)
            for trial, color in zip(range(1, num_all_trials+1), mcolors.TABLEAU_COLORS):
                df = df_all_subjects[f"ID{ID}"][condition][f"agents{num_agents}_tri{trial}"]
                for x, y in zip(df['posX'], df['posY']):
                    ax.scatter(x, y, color=color, alpha=.5)
                pbar.update(1)
            ax.set_title(f"ID{ID}_{condition}_agents{num_agents}")
    print("plotting...")
    plt.show()

# %% 
