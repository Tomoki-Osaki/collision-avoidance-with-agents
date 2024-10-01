import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from tqdm import tqdm

def show_all(obj):
    for i in obj: print(i)
    
    
def make_start_from_UL(df):
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


def line_equation_to_goal(x1, y1, x2=880, y2=880):
    xmin, xmax = x1-20, x1+20
    ymin, ymax = y1-20, y1+20
    
    if x1 == 880:
        return x1, ymax
    if y1 == 880:
        return xmax, y1
            
    slope = (x2 - x1) / (y2 - y1)
    y = slope * (xmax - x1) + y1
    x = (ymax - y1) / slope + x1
    y, x = np.round(y, 3), np.round(x, 3)

    if y <= ymax and y >= ymin:
        return xmax, y
    else: 
        return x, ymax
    
    
def calc_ideal_positions(df):
    posIdeal = df.apply(lambda df: line_equation_to_goal(df["posX"], df["posY"]), axis=1)
    posIdeal.reset_index(drop=True, inplace=True)
    idealX, idealY = [], []
    for i in range(len(posIdeal)):
        idealX.append(posIdeal[i][0])
        idealY.append(posIdeal[i][1])
    df['idealNextX'] = idealX
    df['idealNextY'] = idealY
    
    return df


def make_dict_containing_all_info(num_of_participants):
    """
    Make a dictionary that contains all participants' records.
    The dictionary's hierarchy is: 
        participant's ID:
            experiment conditions (control, urgent, nonurgent, omoiyari):
                trial (trial number and number of agents)
                
    ex. df_1_omoiyari_5_1 = df_all_participants["ID1"]["omoiyari"]["agents5_tri1"]
    """
    pd.options.mode.chained_assignment = None # otherwise there'll be many warnings
    
    exp_conditions = ["omoiyari", "urgent", "nonurgent", "control"]
    
    df_all_participants = {}
    for subjectID in range(1, num_of_participants+1):
        df_all_participants[f"ID{subjectID}"] = {}
    
    with tqdm(total=len(exp_conditions)*num_of_participants) as pbar:
        for condition in exp_conditions:
            for ID in range(1, 30):
            
                df = pd.read_csv(f'04_RawData/{ID}_{condition}.csv')
                
                df_5 = df.query('type.str.contains("05")')
                df_10 = df.query('type.str.contains("10")')
                df_20 = df.query('type.str.contains("20")')
                
                pd.set_option('display.max_columns', None)
                
                trialnum5 = list(set(df_5['trial']))
                trialnum10 = list(set(df_10['trial']))
                trialnum20 = list(set(df_20['trial']))
                assert len(trialnum5) == len(trialnum10) == len(trialnum20), "all trials must have same lengths"
                
                dfs_per_trials = {}
                for trial in range(len(trialnum5)):
                    df_5_tri = df_5.query("trial == @trialnum5[@trial]")
                    df_5_tri = make_start_from_UL(df_5_tri)
                    df_5_tri = calc_ideal_positions(df_5_tri)
                    df_10_tri = df_10.query("trial == @trialnum10[@trial]")
                    df_10_tri = make_start_from_UL(df_10_tri)
                    df_10_tri = calc_ideal_positions(df_10_tri)
                    df_20_tri = df_20.query("trial == @trialnum20[@trial]")
                    df_20_tri = make_start_from_UL(df_20_tri)
                    df_20_tri = calc_ideal_positions(df_20_tri)
                    
                    dfs_per_trials[f"agents5_tri{trial+1}"] = df_5_tri
                    dfs_per_trials[f"agents10_tri{trial+1}"] = df_10_tri
                    dfs_per_trials[f"agents20_tri{trial+1}"] = df_20_tri
                                    
                df_all_participants[f"ID{ID}"][f"{condition}"] = dfs_per_trials
                pbar.update(1)

    return df_all_participants


def plot_traj_per_trials(df_all_participants, ID, conditions, num_agents):
    """
    Plot each trial's trajectory in different axes(2x4).
    """
    fig = plt.figure(figsize=(12, 6))
    for trial in range(1, 9):
        df = df_all_participants[f"ID{ID}"][conditions][f"agents{num_agents}_tri{trial}"]
        ax = fig.add_subplot(2, 4, trial)
        for x, y in zip(df['posX'], df['posY']):
            ax.scatter(x, y, color="blue", alpha=.5)
        ax.set_title(f"tri{trial}")
    plt.suptitle(f"ID{ID}_{conditions}_agents{num_agents}")
    plt.show()
    
    
def plot_traj_all_trials(df_all_participants, ID, conditions, num_agents):
    """
    Plot each trial's trajectory in one axes.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    for trial, color in zip(range(1, 9), mcolors.BASE_COLORS):
        df = df_all_participants[f"ID{ID}"][conditions][f"agents{num_agents}_tri{trial}"]
        for x, y in zip(df['posX'], df['posY']):
            ax.scatter(x, y, color=color, alpha=.5)
    ax.set_title(f"ID{ID}_{conditions}_agents{num_agents}")
    plt.show()
    

def plot_traj_compare_conds(df_all_participants, ID, num_agents):
    """
    Plot each trial's trajectory for all experiment conditions.
    Figure has 1x3 axes.
    """
    assert num_agents in [5, 10, 20], "num_agents must be 5, 10, or 20"
    fig = plt.figure(figsize=(12, 6))
    conditions = ["urgent", "nonurgent", "omoiyari"]
    for i, condition in enumerate(conditions):
        ax = fig.add_subplot(1, 3, i+1)
        for trial, color in zip(range(1, 9), mcolors.BASE_COLORS):
            df = df_all_participants[f"ID{ID}"][condition][f"agents{num_agents}_tri{trial}"]
            for x, y in zip(df['posX'], df['posY']):
                ax.scatter(x, y, color=color, alpha=.4)
        ax.set_title(f"ID{ID}_{condition}_agents{num_agents}")
    plt.show()

