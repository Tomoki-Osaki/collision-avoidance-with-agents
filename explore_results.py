import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
folder = 'C:/Users/ootmo/OneDrive/Desktop/修論/トヨタ_共同P/collision-avoidance-with-agents/シミュレーション結果_コピー/'
folder = 'C:/Users/Tomoki/OneDrive/Documents/共同P_論文/collision-avoidance-with-agents/シミュレーション結果_コピー/'    
folder = 'C:/Users/Tomoki/OneDrive/Documents/共同P_論文/collision-avoidance-with-agents/simulation_results/'

def merge_result_seed_files(folder: str, agent: int, avoid_vec: float):
    """
    create one result dataframe which contains the simulation results of all different seeds.
    """
    df = pd.DataFrame()
    for i in range(0, 11):
        tmp = pd.read_csv(folder+f'agt{agent}_avoidvec{int(avoid_vec*10)}px_dynper0{i}.csv')
        df = pd.concat([df, tmp], ignore_index=True)
    df.drop('Unnamed: 0', axis=1, inplace=True) # seed
    df['avoid_vec_px'] = df['simple_avoid_vec'] * 50
    return df
    
# %% agent 25
df_agt25_avoid1 = merge_result_seed_files(folder, 25, 1)
df_agt25_avoid1_mean = df_agt25_avoid1.groupby('dynamic_percent', as_index=False).mean()
df_agt25_avoid2 = merge_result_seed_files(folder, 25, 2)
df_agt25_avoid2_mean = df_agt25_avoid2.groupby('dynamic_percent', as_index=False).mean()
df_agt25_avoid3 = merge_result_seed_files(folder, 25, 3)
df_agt25_avoid3_mean = df_agt25_avoid3.groupby('dynamic_percent', as_index=False).mean()
df_agt25_avoid4 = merge_result_seed_files(folder, 25, 4)
df_agt25_avoid4_mean = df_agt25_avoid4.groupby('dynamic_percent', as_index=False).mean()
df_agt25_avoid5 = merge_result_seed_files(folder, 25, 5)
df_agt25_avoid5_mean = df_agt25_avoid5.groupby('dynamic_percent', as_index=False).mean()
df_agt25_mean = pd.concat([df_agt25_avoid1_mean, 
                           df_agt25_avoid2_mean, 
                           df_agt25_avoid3_mean, 
                           df_agt25_avoid4_mean, 
                           df_agt25_avoid5_mean])

# %% agent 50
df_agt50_avoid1 = merge_result_seed_files(folder, 50, 1)
df_agt50_avoid1_mean = df_agt50_avoid1.groupby('dynamic_percent', as_index=False).mean()

df_agt50_avoid15 = merge_result_seed_files(folder, 50, 1.5)
df_agt50_avoid15.drop('seed', axis=1, inplace=True)
df_agt50_avoid15_mean = df_agt50_avoid15.groupby('dynamic_percent', as_index=False).mean()

df_agt50_avoid16 = merge_result_seed_files(folder, 50, 1.6)
df_agt50_avoid16_mean = df_agt50_avoid16.groupby('dynamic_percent', as_index=False).mean()

df_agt50_avoid17 = merge_result_seed_files(folder, 50, 1.7)
df_agt50_avoid17_mean = df_agt50_avoid17.groupby('dynamic_percent', as_index=False).mean()

df_agt50_avoid18 = merge_result_seed_files(folder, 50, 1.8)
df_agt50_avoid18_mean = df_agt50_avoid18.groupby('dynamic_percent', as_index=False).mean()

df_agt50_avoid19 = merge_result_seed_files(folder, 50, 1.9)
df_agt50_avoid19_mean = df_agt50_avoid19.groupby('dynamic_percent', as_index=False).mean()

df_agt50_avoid2 = merge_result_seed_files(folder, 50, 2)
df_agt50_avoid2_mean = df_agt50_avoid2.groupby('dynamic_percent', as_index=False).mean()

df_agt50_avoid21 = merge_result_seed_files(folder, 50, 2.1)
df_agt50_avoid21_mean = df_agt50_avoid21.groupby('dynamic_percent', as_index=False).mean()

df_agt50_avoid22 = merge_result_seed_files(folder, 50, 2.2)
df_agt50_avoid22_mean = df_agt50_avoid22.groupby('dynamic_percent', as_index=False).mean()

df_agt50_avoid23 = merge_result_seed_files(folder, 50, 2.3)
df_agt50_avoid23_mean = df_agt50_avoid23.groupby('dynamic_percent', as_index=False).mean()

df_agt50_avoid24 = merge_result_seed_files(folder, 50, 2.4)
df_agt50_avoid24_mean = df_agt50_avoid24.groupby('dynamic_percent', as_index=False).mean()

df_agt50_avoid25 = merge_result_seed_files(folder, 50, 2.5)
df_agt50_avoid25_mean = df_agt50_avoid25.groupby('dynamic_percent', as_index=False).mean()

df_agt50_avoid3 = merge_result_seed_files(folder, 50, 3)
df_agt50_avoid3_mean = df_agt50_avoid3.groupby('dynamic_percent', as_index=False).mean()

df_agt50_avoid4 = merge_result_seed_files(folder, 50, 4)
df_agt50_avoid4_mean = df_agt50_avoid4.groupby('dynamic_percent', as_index=False).mean()

df_agt50_avoid5 = merge_result_seed_files(folder, 50, 5)
df_agt50_avoid5_mean = df_agt50_avoid5.groupby('dynamic_percent', as_index=False).mean()

df_agt50_mean = pd.concat([df_agt50_avoid1_mean, 
                           df_agt50_avoid2_mean, 
                           df_agt50_avoid3_mean, 
                           df_agt50_avoid4_mean, 
                           df_agt50_avoid5_mean])

# %% agent 100
df_agt100_avoid1 = merge_result_seed_files(folder, 100, 1)
df_agt100_avoid1_mean = df_agt100_avoid1.groupby('dynamic_percent', as_index=False).mean()

df_agt100_avoid2 = merge_result_seed_files(folder, 100, 2)
df_agt100_avoid2_mean = df_agt100_avoid2.groupby('dynamic_percent', as_index=False).mean()

df_agt100_avoid3 = merge_result_seed_files(folder, 100, 3)
df_agt100_avoid3_mean = df_agt100_avoid3.groupby('dynamic_percent', as_index=False).mean()

df_agt100_avoid4 = merge_result_seed_files(folder, 100, 4)
df_agt100_avoid4_mean = df_agt100_avoid4.groupby('dynamic_percent', as_index=False).mean()

df_agt100_avoid5 = merge_result_seed_files(folder, 100, 5)
df_agt100_avoid5_mean = df_agt100_avoid5.groupby('dynamic_percent', as_index=False).mean()

df_agt100_mean = pd.concat([df_agt100_avoid1_mean,
                            df_agt100_avoid2_mean,
                            df_agt100_avoid3_mean,
                            df_agt100_avoid4_mean,
                            df_agt100_avoid5_mean])

# %%
s = 50
plt.rcParams['font.family'] = "MS Gothic"

fig, ax = plt.subplots()
ax.scatter(df_agt50_avoid1_mean['time'][0], df_agt50_avoid1_mean['collision'][0], 
           s=s, label='1px')
ax.scatter(df_agt50_avoid2_mean['time'][0], df_agt50_avoid2_mean['collision'][0], 
           s=s, label='2px')
ax.scatter(df_agt50_avoid3_mean['time'][0], df_agt50_avoid3_mean['collision'][0], 
           s=s, label='3px')
ax.scatter(df_agt50_avoid4_mean['time'][0], df_agt50_avoid4_mean['collision'][0], 
           s=s, label='4px')
ax.scatter(df_agt50_avoid5_mean['time'][0], df_agt50_avoid5_mean['collision'][0], 
           s=s, label='5px')
ax.scatter(df_agt50_avoid5_mean['time'][1.0], df_agt50_avoid5_mean['collision'][1.0], 
           s=s, label='動的回避')
ax.set_xlabel('完了時間 (ステップ)')
ax.set_ylabel('妨\n害\n量\n(回)', rotation=0)
ax.yaxis.set_label_coords(-0.1, 0.5)
ax.legend(title='回避ベクトル')
ax.grid()
plt.tight_layout()
plt.show()


fig, ax = plt.subplots()

def plot_continuous(df_mean: pd.DataFrame, color: str):
    size = 50
    for i in np.arange(0, 1.1, 0.1): 
        idx = np.round(i, 1)
        ax.scatter(df_mean['time'][idx], 
                   df_mean['collision'][idx],
                   s=size, color=color)
        size *= 0.8
        
fig, ax = plt.subplots()
plot_continuous(df_agt25_avoid1_mean, color='blue')
plot_continuous(df_agt25_avoid2_mean, color='red')
plot_continuous(df_agt25_avoid3_mean, color='green')
plot_continuous(df_agt25_avoid4_mean, color='yellow')
plot_continuous(df_agt25_avoid5_mean, color='black')
plot_continuous(df_agt50_avoid1_mean, color='blue')
plot_continuous(df_agt50_avoid2_mean, color='red')
plot_continuous(df_agt50_avoid3_mean, color='green')
plot_continuous(df_agt50_avoid4_mean, color='yellow')
plot_continuous(df_agt50_avoid5_mean, color='black')
plot_continuous(df_agt100_avoid1_mean, color='blue')
plot_continuous(df_agt100_avoid2_mean, color='red')
plot_continuous(df_agt100_avoid3_mean, color='green')
plot_continuous(df_agt100_avoid4_mean, color='yellow')
plot_continuous(df_agt100_avoid5_mean, color='black')
plt.show()

