"""
1.  show_all
2.  make_start_from_UL
3.  _calc_ideal_positions
4.  add_cols_ideal_positions
5.  calc_dist_actual_ideal
6.  drop_unnecessary_cols
7.  _calc_distance
8.  add_cols_dist_others
9.  calc_closest_others
10. preprocess
11. make_empty_hierarchical_df
12. make_dict_of_all_info
13. make_df_trial
13. plot_traj_per_trials
14. plot_traj_compare_conds
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
sns.set_theme()
from tqdm import tqdm
from typing import Literal

# %% Global varibales
SUBJECTS = [subject for subject in range(1, 30)]
CONDITIONS = ['urgent', 'nonurgent', 'omoiyari']
AGENTS = [agent for agent in range(1, 21)]
NUM_AGENTS = [5, 10, 20]
TRIALS = [trial for trial in range(1, 9)]

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
    df.drop("Unnamed: 117",axis=1, inplace=True)
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
    df = calc_dist_actual_ideal(df)
    df = drop_unnecessary_cols(df)
    df = add_cols_dist_others(df)
    df = calc_closest_others(df)
    
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

# %%
def make_df_trial(df_all: pd.DataFrame, 
                  ID: int, 
                  condition: Literal['urgent', 'nonurgent', 'omoiyari'], 
                  num_agents: int, 
                  trial: int) -> pd.DataFrame:
    
    df = df_all[f'ID{ID}'][condition][f"agents{num_agents}"][f"trial{trial}"]
    
    return df
    
# %%
def plot_traj_per_trials(df_all: pd.DataFrame, 
                         ID: int, 
                         conditions: Literal["urgent", "nonurgent", "omoiyari"], 
                         num_agents: Literal[5, 10, 20]) -> None:
    """
    Plot each trial's trajectory in different axes(2x4).
    """
    fig = plt.figure(figsize=(12, 6), tight_layout=True)
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
    fig = plt.figure(figsize=(12, 6), tight_layout=True)
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
