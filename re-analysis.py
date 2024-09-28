# control, urgent, nonurgent, omoiyari
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def show_all(obj):
    for i in obj: print(i)
    
def make_start_from_UL(df):
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

def make_dict_containing_all_info():
    pd.options.mode.chained_assignment = None # otherwise there'll be many warnings
    
    exp_conditions = ["omoiyari", "urgent", "nonurgent", "control"]
    df_all_participants = {}
    for subjectID in range(1, 30):
        df_all_participants[f"ID_{subjectID}"] = {}
        
    for condition in exp_conditions:
        for i in range(1, 30):
        
            df = pd.read_csv(f'04_RawData/{i}_{condition}.csv')
            
            df_5 = df.query('type.str.contains("05")')
            df_10 = df.query('type.str.contains("10")')
            df_20 = df.query('type.str.contains("20")')
            
            tri_num_5 = list(set(df_5['trial']))
            tri_num_10 = list(set(df_10['trial']))
            tri_num_20 = list(set(df_20['trial']))
            assert len(tri_num_5) == len(tri_num_10) == len(tri_num_20), "all trials must have same lengths"
            
            dfs_per_trials = {}
            for j in range(len(tri_num_5)):
                df_5_tri = df_5.query("trial == @tri_num_5[@j]")
                df_5_tri = make_start_from_UL(df_5_tri)
                df_10_tri = df_10.query("trial == @tri_num_10[@j]")
                df_10_tri = make_start_from_UL(df_10_tri)
                df_20_tri = df_20.query("trial == @tri_num_20[@j]")
                df_20_tri = make_start_from_UL(df_20_tri)
                
                dfs_per_trials[f"agents5_tri{j+1}"] = df_5_tri
                dfs_per_trials[f"agents10_tri{j+1}"] = df_10_tri
                dfs_per_trials[f"agents20_tri{j+1}"] = df_20_tri
        
            df_all_participants[f"ID_{i}"][f"{condition}"] = dfs_per_trials

    return df_all_participants

df_all_participants = make_dict_containing_all_info()

# df_id_conditions_NumOfAgents_Trialnumber
df1omoiyari51 = df_all_participants["ID_1"]["omoiyari"]["agents5_tri1"]

def plot_all_trials(ID, conditions, num_of_agents):
    fig = plt.figure(figsize=(12, 8))
    for trial in range(1, 9):
        df = df_all_participants[f"ID_{ID}"][conditions][f"agents{num_of_agents}_tri{trial}"]
        ax = fig.add_subplot(2, 4, trial)
        for x, y in zip(df['posX'], df['posY']):
            ax.scatter(x, y, color="blue", alpha=.5)
        ax.set_title(f"tri{trial}")
    plt.suptitle(f"ID_{ID}_{conditions}_agents{num_of_agents}")
    plt.show()

plot_all_trials(1, "omoiyari", 20)
plot_all_trials(1, "urgent", 20)

