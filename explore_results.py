import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = "MS Gothic"

# %%
folder = 'シミュレーション結果_コピー/'    
folder = 'simulation_results/'

def merge_result_seed_files(folder: str, agent: int, avoid_vec: float):
    """
    create one result dataframe which contains the simulation results of all different seeds.
    """
    df = pd.DataFrame()
    for i in range(0, 11):
        tmp = pd.read_csv(folder+f'agt{agent}_avoidvec{int(avoid_vec*10)}px_dynper0{i}.csv')
        df = pd.concat([df, tmp], ignore_index=True)
    df.drop('Unnamed: 0', axis=1, inplace=True) # drop the column of seed
    df['avoid_vec_px'] = df['simple_avoid_vec'] * 50
    return df


def plot_values_transitions(df_dict: dict, variable: str, vectors: iter,
                            color='blue', figsize=(30, 6), 
                            xticks=[i for i in np.arange(0, 1.1, 0.2)]):
    figsize = (30, 6)
    fig, ax = plt.subplots(1, len(df_dict), figsize=figsize, sharex=True, sharey=True)
    for i, vec in enumerate(vectors):
        vec = np.round(vec, 1)
        ax[i].errorbar(df_dict[vec]['mean']['dynamic_percent'], df_dict[vec]['mean'][variable], 
                       yerr=df_dict[vec]['std'][variable], 
                       capsize=5, marker='o', c=color)
        ax[i].grid()
    plt.xticks(ticks=xticks)
    plt.show()
    
    
def plot_continuous(df_mean, color, label):
    alpha = 0.5
    size = 35
    for i in range(11):
        ax.scatter(df_mean['time'][i],
                    df_mean['collision'][i],
                    alpha=alpha, color=color, s=size)
        size *= 1.35
        if i == 0:
            ax.scatter(df_mean['time'][i],
                       df_mean['collision'][i],
                       alpha=alpha, color=color, s=size, label=label)
            ax.text(df_mean['time'][i]+0.01,
                       df_mean['collision'][i]+0.01, label)
    ax.legend(title='回避ベクトル')
    ax.grid()   
    
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

# %% agent 50
df_agt50_avoid1 = merge_result_seed_files(folder, 50, 1)
df_agt50_avoid1_mean = df_agt50_avoid1.groupby('dynamic_percent', as_index=False).mean()
df_agt50_avoid1_std = df_agt50_avoid1.groupby('dynamic_percent', as_index=False).std()

df_agt50_avoid2 = merge_result_seed_files(folder, 50, 2)
df_agt50_avoid2_mean = df_agt50_avoid2.groupby('dynamic_percent', as_index=False).mean()
df_agt50_avoid2_std = df_agt50_avoid2.groupby('dynamic_percent', as_index=False).std()

df_agt50_avoid3 = merge_result_seed_files(folder, 50, 3)
df_agt50_avoid3_mean = df_agt50_avoid3.groupby('dynamic_percent', as_index=False).mean()
df_agt50_avoid3_std = df_agt50_avoid3.groupby('dynamic_percent', as_index=False).std()

df_agt50_avoid4 = merge_result_seed_files(folder, 50, 4)
df_agt50_avoid4_mean = df_agt50_avoid4.groupby('dynamic_percent', as_index=False).mean()
df_agt50_avoid4_std = df_agt50_avoid4.groupby('dynamic_percent', as_index=False).std()

df_agt50_avoid5 = merge_result_seed_files(folder, 50, 5)
df_agt50_avoid5_mean = df_agt50_avoid5.groupby('dynamic_percent', as_index=False).mean()
df_agt50_avoid5_std = df_agt50_avoid5.groupby('dynamic_percent', as_index=False).std()

df_agt50 = {1: {'mean': df_agt50_avoid1_mean, 'std': df_agt50_avoid1_std},
            2: {'mean': df_agt50_avoid2_mean, 'std': df_agt50_avoid2_std},
            3: {'mean': df_agt50_avoid3_mean, 'std': df_agt50_avoid3_std},
            4: {'mean': df_agt50_avoid4_mean, 'std': df_agt50_avoid4_std},
            5: {'mean': df_agt50_avoid5_mean, 'std': df_agt50_avoid5_std}}

# %% agent 50 1-5px plot collision
plot_values_transitions(df_agt50, 'time', range(1,5), color='tab:blue')
plot_values_transitions(df_agt50, 'collision', range(1,5), color='tab:orange')

# %% agent50 1.5-2.5px
df_agt50_avoid16 = merge_result_seed_files(folder, 50, 1.6)
df_agt50_avoid16_mean = df_agt50_avoid16.groupby('dynamic_percent', as_index=False).mean()
df_agt50_avoid16_std = df_agt50_avoid16.groupby('dynamic_percent', as_index=False).std()

df_agt50_avoid17 = merge_result_seed_files(folder, 50, 1.7)
df_agt50_avoid17_mean = df_agt50_avoid17.groupby('dynamic_percent', as_index=False).mean()
df_agt50_avoid17_std = df_agt50_avoid17.groupby('dynamic_percent', as_index=False).std()

df_agt50_avoid18 = merge_result_seed_files(folder, 50, 1.8)
df_agt50_avoid18_mean = df_agt50_avoid18.groupby('dynamic_percent', as_index=False).mean()
df_agt50_avoid18_std = df_agt50_avoid18.groupby('dynamic_percent', as_index=False).std()

df_agt50_avoid19 = merge_result_seed_files(folder, 50, 1.9)
df_agt50_avoid19_mean = df_agt50_avoid19.groupby('dynamic_percent', as_index=False).mean()
df_agt50_avoid19_std = df_agt50_avoid19.groupby('dynamic_percent', as_index=False).std()

df_agt50_avoid21 = merge_result_seed_files(folder, 50, 2.1)
df_agt50_avoid21_mean = df_agt50_avoid21.groupby('dynamic_percent', as_index=False).mean()
df_agt50_avoid21_std = df_agt50_avoid21.groupby('dynamic_percent', as_index=False).std()

df_agt50_avoid22 = merge_result_seed_files(folder, 50, 2.2)
df_agt50_avoid22_mean = df_agt50_avoid22.groupby('dynamic_percent', as_index=False).mean()
df_agt50_avoid22_std = df_agt50_avoid22.groupby('dynamic_percent', as_index=False).std()

df_agt50_avoid23 = merge_result_seed_files(folder, 50, 2.3)
df_agt50_avoid23_mean = df_agt50_avoid23.groupby('dynamic_percent', as_index=False).mean()
df_agt50_avoid23_std = df_agt50_avoid23.groupby('dynamic_percent', as_index=False).std()

# df_agt50_avoid15 = merge_result_seed_files(folder, 50, 1.5)
# df_agt50_avoid15_mean = df_agt50_avoid15.groupby('dynamic_percent', as_index=False).mean()

# df_agt50_avoid24 = merge_result_seed_files(folder, 50, 2.4)
# df_agt50_avoid24_mean = df_agt50_avoid24.groupby('dynamic_percent', as_index=False).mean()

# df_agt50_avoid25 = merge_result_seed_files(folder, 50, 2.5)
# df_agt50_avoid25_mean = df_agt50_avoid25.groupby('dynamic_percent', as_index=False).mean()

df_agt50_1623 = {1.6: {'mean': df_agt50_avoid16_mean, 'std': df_agt50_avoid16_std},
                 1.7: {'mean': df_agt50_avoid17_mean, 'std': df_agt50_avoid17_std},
                 1.8: {'mean': df_agt50_avoid18_mean, 'std': df_agt50_avoid18_std},
                 1.9: {'mean': df_agt50_avoid19_mean, 'std': df_agt50_avoid19_std},
                 2.0: {'mean': df_agt50_avoid2_mean, 'std': df_agt50_avoid2_std},
                 2.1: {'mean': df_agt50_avoid21_mean, 'std': df_agt50_avoid21_std},
                 2.2: {'mean': df_agt50_avoid22_mean, 'std': df_agt50_avoid22_std},
                 2.3: {'mean': df_agt50_avoid23_mean, 'std': df_agt50_avoid23_std}}

# %%
plot_values_transitions(df_agt50_1623, 'time', np.arange(1.6, 2.4, 0.1), 
                        color='tab:blue', figsize=(34, 5))
plot_values_transitions(df_agt50_1623, 'collision', np.arange(1.6, 2.4, 0.1), color='tab:orange')

# %% agent 100
df_agt100_avoid1 = merge_result_seed_files(folder, 100, 1)
df_agt100_avoid1_mean = df_agt100_avoid1.groupby('dynamic_percent', as_index=False).mean()
df_agt100_avoid1_std = df_agt100_avoid1.groupby('dynamic_percent', as_index=False).std()

df_agt100_avoid2 = merge_result_seed_files(folder, 100, 2)
df_agt100_avoid2_mean = df_agt100_avoid2.groupby('dynamic_percent', as_index=False).mean()
df_agt100_avoid2_std = df_agt100_avoid2.groupby('dynamic_percent', as_index=False).std()

df_agt100_avoid3 = merge_result_seed_files(folder, 100, 3)
df_agt100_avoid3_mean = df_agt100_avoid3.groupby('dynamic_percent', as_index=False).mean()
df_agt100_avoid3_std = df_agt100_avoid3.groupby('dynamic_percent', as_index=False).std()

df_agt100_avoid4 = merge_result_seed_files(folder, 100, 4)
df_agt100_avoid4_mean = df_agt100_avoid4.groupby('dynamic_percent', as_index=False).mean()
df_agt100_avoid4_std = df_agt100_avoid4.groupby('dynamic_percent', as_index=False).std()

df_agt100_avoid5 = merge_result_seed_files(folder, 100, 5)
df_agt100_avoid5_mean = df_agt100_avoid5.groupby('dynamic_percent', as_index=False).mean()
df_agt100_avoid5_std = df_agt100_avoid5.groupby('dynamic_percent', as_index=False).std()

df_agt100 = {1: {'mean': df_agt100_avoid1_mean, 'std': df_agt100_avoid1_std},
             2: {'mean': df_agt100_avoid2_mean, 'std': df_agt100_avoid2_std},
             3: {'mean': df_agt100_avoid3_mean, 'std': df_agt100_avoid3_std},
             4: {'mean': df_agt100_avoid4_mean, 'std': df_agt100_avoid4_std},
             5: {'mean': df_agt100_avoid5_mean, 'std': df_agt100_avoid5_std}}

# %%
plot_values_transitions(df_agt100, 'time', range(1,5), color='tab:blue')
plot_values_transitions(df_agt100, 'collision', range(1,5), color='tab:orange')
 
# %%
path50 = "C:/Users/Tomoki/Downloads/motiduki_agents50.csv"
path50 = "C:/Users/ootmo/Downloads/motiduki_agents50.csv"
df50 = pd.read_csv(path50)

df50 = df50[['回避ベクトル', '完了時間', '妨害量']]
df50.rename({'回避ベクトル': 'avoid_vec', '完了時間': 'time', '妨害量': 'collision'}, 
            axis='columns', inplace=True)
df50simple = df50.iloc[:-1, :]
df50simple = df50simple.astype('float')
df50dynamic = df50.iloc[-1, :]

plt.scatter(df50simple.time, df50simple.collision)
plt.scatter(df50dynamic.time, df50dynamic.collision)


path100 = "C:/Users/Tomoki/Downloads/motiduki_agents100.csv"
path100 = "C:/Users/ootmo/Downloads/motiduki_agents100.csv"
df100 = pd.read_csv(path100)

df100 = df100[['回避ベクトル', '完了時間', '妨害量']]
df100.rename({'回避ベクトル': 'avoid_vec', '完了時間': 'time', '妨害量': 'collision'}, 
             axis='columns', inplace=True)
df100simple = df100.iloc[:-1, :]
df100simple = df100simple.astype('float')
df100dynamic = df100.iloc[-1, :]

plt.scatter(df100simple.time, df100simple.collision)
plt.scatter(df100dynamic.time, df100dynamic.collision)

# %% colors and annotations
annots = ['0', '.1', '.2', '.3', '.4', '.5', '.6', '.7', '.8', '.9', '1.']
grayalpha = 0.3
alpha = 0.7
size = 60
txt_far = 0.05
figsize = (18, 9)

# %%
fig, ax = plt.subplots(figsize=figsize)
for i, j in enumerate(df50simple.avoid_vec):
    j = np.round(j, 2)
    if i == 0:
        ax.scatter(df50simple.time[i], df50simple.collision[i], color='tab:gray', alpha=0.7, s=30,
                   label='1-5px(0.1px刻み)')
    else:
        ax.scatter(df50simple.time[i], df50simple.collision[i], color='tab:gray', alpha=0.7, s=30)
    if j == 1.0 or j == 2.0 or j == 3.0 or j == 4.0 or j == 5.0:
        ax.text(df50simple.time[i]+txt_far, 
                df50simple.collision[i]+txt_far,
                f'{int(j)}px')
        
ax.scatter(df_agt50_avoid1_mean['time'][10],
           df_agt50_avoid1_mean['collision'][10],
           color='black', s=size,
           label='動的回避')
ax.legend(title='回避ベクトルの大きさ')
ax.grid()

# %%
def plot_visible_transitions(df_dict, vec, color='tab:orange'):
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(df50simple.time, df50simple.collision, color='tab:gray', alpha=grayalpha, s=30,
               label='1-5px(0.1px刻み)')
    for i, annot in zip(range(11), annots):
        
        if i == 0:
            ax.scatter(df_dict[vec]['mean']['time'][i],
                       df_dict[vec]['mean']['collision'][i],
                       color=color, s=size, label=f'{vec}px', alpha=alpha)
        else:
            ax.scatter(df_dict[vec]['mean']['time'][i],
                       df_dict[vec]['mean']['collision'][i],
                       color=color, s=size, alpha=alpha)
            
        ax.text(df_dict[vec]['mean']['time'][i]+txt_far,
                df_dict[vec]['mean']['collision'][i]+txt_far, 
                annot, size='small')
    ax.grid()
    ax.legend(title='回避ベクトルの大きさ')
    plt.show()

# %% 
plot_visible_transitions(df_agt50, 1, color='tab:orange')
plot_visible_transitions(df_agt50, 2, color='tab:green')
plot_visible_transitions(df_agt50, 3, color='tab:red')
plot_visible_transitions(df_agt50, 4, color='tab:blue')
plot_visible_transitions(df_agt50, 5, color='tab:brown')

# %%
figsize = (18, 9)
fig, ax = plt.subplots(figsize=figsize)
plot_continuous(df_agt50_avoid1_mean, color='tab:orange', label='1px')
plot_continuous(df_agt50_avoid2_mean, color='tab:green', label='2px')
plot_continuous(df_agt50_avoid3_mean, color='tab:red', label='3px')
plot_continuous(df_agt50_avoid4_mean, color='tab:blue', label='4px')
plot_continuous(df_agt50_avoid5_mean, color='tab:brown', label='5px')
plt.show()

fig, ax = plt.subplots(figsize=figsize)
plot_continuous(df_agt100_avoid1_mean, color='tab:orange', label='1px')
plot_continuous(df_agt100_avoid2_mean, color='tab:green', label='2px')
plot_continuous(df_agt100_avoid3_mean, color='tab:red', label='3px')
plot_continuous(df_agt100_avoid4_mean, color='tab:blue', label='4px')
plot_continuous(df_agt100_avoid5_mean, color='tab:brown', label='5px')
plt.show()

