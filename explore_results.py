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
    for i in range(11): 
        ax.scatter(df_mean['time'][i], 
                   df_mean['collision'][i],
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
 
# %%
path50 = "C:/Users/Tomoki/Downloads/motiduki_agents50.csv"
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
df100 = pd.read_csv(path100)

df100 = df100[['回避ベクトル', '完了時間', '妨害量']]
df100.rename({'回避ベクトル': 'avoid_vec', '完了時間': 'time', '妨害量': 'collision'}, 
             axis='columns', inplace=True)
df100simple = df100.iloc[:-1, :]
df100simple = df100simple.astype('float')
df100dynamic = df100.iloc[-1, :]

plt.scatter(df100simple.time, df100simple.collision)
plt.scatter(df100dynamic.time, df100dynamic.collision)

# %% continuous plots for the presentation
plt.rcParams['font.size'] = 18
figsize = (18, 9)

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
    ax.set_xlabel('完了時間(ステップ)')
    ax.set_ylabel('妨\n害\n量\n(回)', rotation=0)
    ax.grid()   

# %%
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
    ax.set_xlabel('完了時間(ステップ)')
    ax.set_ylabel('妨\n害\n量\n(回)', rotation=0)
    ax.grid()   
plt.rcParams['font.size'] = 18
figsize = (18, 9)


# %% colors and annotations
cdict = {
    'blue': "#377eb8",  # 青
    'orange': "#ff7f00",  # オレンジ
    'green': "#4daf4a",  # 緑
    'red': "#e41a1c",  # 赤
    'purple': "#984ea3",  # 紫
    'yellow': "#ffcc00",  # 黄
    'brown': "#a65628",   # 茶
    'gray': "#808080" # dark gray for low importance
}

annots = ['0', '.1', '.2', '.3', '.4', '.5', '.6', '.7', '.8', '.9', '1.']
grayalpha = 0.3
alpha = 0.7
size = 60
txt_far = 0.05

# %% plot increasing of dynamic agents    
# all in one 
fig, ax = plt.subplots(figsize=figsize)
ax.scatter(df50simple.time, df50simple.collision, color=cdict['gray'], alpha=grayalpha, s=30,
           label='1-5px(0.1px刻み)')
for i, annot in zip(range(11), annots):
    
    color = cdict['orange']
    if i == 0:
        ax.scatter(df_agt50_avoid1_mean['time'][i],
                   df_agt50_avoid1_mean['collision'][i],
                   color=color, s=size, label='1px', alpha=alpha)
    else:
        ax.scatter(df_agt50_avoid1_mean['time'][i],
                   df_agt50_avoid1_mean['collision'][i],
                   color=color, s=size, alpha=alpha)
        
    ax.text(df_agt50_avoid1_mean['time'][i]+txt_far,
            df_agt50_avoid1_mean['collision'][i]+txt_far, 
            annot, size='small')
    
    color = cdict['green']
    if i == 0:
        ax.scatter(df_agt50_avoid2_mean['time'][i],
                   df_agt50_avoid2_mean['collision'][i],
                   color=color, s=size, label='2px', alpha=alpha)
    else:
        ax.scatter(df_agt50_avoid2_mean['time'][i],
                    df_agt50_avoid2_mean['collision'][i],
                    color=cdict['green'], s=60, alpha=alpha)
    ax.text(df_agt50_avoid2_mean['time'][i]+txt_far,
            df_agt50_avoid2_mean['collision'][i]+txt_far, 
            annot, size='small')
    
    color = cdict['brown']
    if i == 0:
        ax.scatter(df_agt50_avoid3_mean['time'][i],
                   df_agt50_avoid3_mean['collision'][i],
                   color=color, s=size, label='3px', alpha=alpha)
    else:
        ax.scatter(df_agt50_avoid3_mean['time'][i],
                    df_agt50_avoid3_mean['collision'][i],
                    color=color, s=size, alpha=alpha)
    ax.text(df_agt50_avoid3_mean['time'][i]+txt_far,
            df_agt50_avoid3_mean['collision'][i]+txt_far, 
            annot, size='small')
    
    color = cdict['purple']
    if i == 0:
        ax.scatter(df_agt50_avoid4_mean['time'][i],
                   df_agt50_avoid4_mean['collision'][i],
                   color=color, s=size, label='4px', alpha=alpha)
    else:
        ax.scatter(df_agt50_avoid4_mean['time'][i],
                   df_agt50_avoid4_mean['collision'][i],
                   alpha=1, color=color, s=60)
    ax.text(df_agt50_avoid4_mean['time'][i]+txt_far,
            df_agt50_avoid4_mean['collision'][i]+txt_far, 
            annot, size='small')
    
    color = cdict['red']
    if i == 0:
        ax.scatter(df_agt50_avoid3_mean['time'][i],
                   df_agt50_avoid3_mean['collision'][i],
                   color=color, s=size, label='5px')
    else:
        ax.scatter(df_agt50_avoid5_mean['time'][i],
                    df_agt50_avoid5_mean['collision'][i],
                    color=color, s=size, alpha=alpha)
    ax.text(df_agt50_avoid5_mean['time'][i]+txt_far,
            df_agt50_avoid5_mean['collision'][i]+txt_far, 
            annot, size='small')
ax.grid()
ax.legend(title='回避ベクトルの大きさ')

# %%
fig, ax = plt.subplots(figsize=figsize)
for i, j in enumerate(df50simple.avoid_vec):
    j = np.round(j, 2)
    if i == 0:
        ax.scatter(df50simple.time[i], df50simple.collision[i], color=cdict['gray'], alpha=0.7, s=30,
                   label='1-5px(0.1px刻み)')
    else:
        ax.scatter(df50simple.time[i], df50simple.collision[i], color=cdict['gray'], alpha=0.7, s=30)
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

# %% 1px 
fig, ax = plt.subplots(figsize=figsize)
ax.scatter(df50simple.time, df50simple.collision, color=cdict['gray'], alpha=grayalpha, s=30,
           label='1-5px(0.1px刻み)')
for i, annot in zip(range(11), annots):
    
    color = cdict['orange']
    if i == 0:
        ax.scatter(df_agt50_avoid1_mean['time'][i],
                   df_agt50_avoid1_mean['collision'][i],
                   color=color, s=size, label='1px', alpha=alpha)
    else:
        ax.scatter(df_agt50_avoid1_mean['time'][i],
                   df_agt50_avoid1_mean['collision'][i],
                   color=color, s=size, alpha=alpha)
        
    ax.text(df_agt50_avoid1_mean['time'][i]+txt_far,
            df_agt50_avoid1_mean['collision'][i]+txt_far, 
            annot, size='small')
ax.grid()
ax.legend(title='回避ベクトルの大きさ')

# %% 2px
fig, ax = plt.subplots(figsize=figsize)
ax.scatter(df50simple.time, df50simple.collision, color=cdict['gray'], alpha=grayalpha, s=30,
           label='1-5px(0.1px刻み)')
for i, annot in zip(range(11), annots):
    
    color = cdict['green']
    if i == 0:
        ax.scatter(df_agt50_avoid2_mean['time'][i],
                   df_agt50_avoid2_mean['collision'][i],
                   color=color, s=size, label='2px', alpha=alpha)
    else:
        ax.scatter(df_agt50_avoid2_mean['time'][i],
                    df_agt50_avoid2_mean['collision'][i],
                    color=cdict['green'], s=size, alpha=alpha)
    ax.text(df_agt50_avoid2_mean['time'][i]+txt_far,
            df_agt50_avoid2_mean['collision'][i]+txt_far, 
            annot, size='small')  
ax.grid()
ax.legend(title='回避ベクトルの大きさ')

# %% 3px
fig, ax = plt.subplots(figsize=figsize)
ax.scatter(df50simple.time, df50simple.collision, color=cdict['gray'], alpha=grayalpha, s=30,
           label='1-5px(0.1px刻み)')
for i, annot in zip(range(11), annots):
    size = 60
    
    color = cdict['brown']
    if i == 0:
        ax.scatter(df_agt50_avoid3_mean['time'][i],
                   df_agt50_avoid3_mean['collision'][i],
                   color=color, s=size, label='3px', alpha=alpha)
    else:
        ax.scatter(df_agt50_avoid3_mean['time'][i],
                    df_agt50_avoid3_mean['collision'][i],
                    color=color, s=size, alpha=alpha)
    ax.text(df_agt50_avoid3_mean['time'][i]+txt_far,
            df_agt50_avoid3_mean['collision'][i]+txt_far, 
            annot, size='small')
    
ax.grid()
ax.legend(title='回避ベクトルの大きさ')

# %% 4px
fig, ax = plt.subplots(figsize=figsize)
ax.scatter(df50simple.time, df50simple.collision, color=cdict['gray'], alpha=grayalpha, s=30,
           label='1-5px(0.1px刻み)')
for i, annot in zip(range(11), annots):
    
    color = cdict['purple']
    if i == 0:
        ax.scatter(df_agt50_avoid4_mean['time'][i],
                   df_agt50_avoid4_mean['collision'][i],
                   color=color, s=size, label='4px', alpha=alpha)
    else:
        ax.scatter(df_agt50_avoid4_mean['time'][i],
                   df_agt50_avoid4_mean['collision'][i],
                   color=color, s=size, alpha=alpha)
    ax.text(df_agt50_avoid4_mean['time'][i]+txt_far,
            df_agt50_avoid4_mean['collision'][i]+txt_far, 
            annot, size='small')
    
ax.grid()
ax.legend(title='回避ベクトルの大きさ')

# %% 5px
fig, ax = plt.subplots(figsize=figsize)
ax.scatter(df50simple.time, df50simple.collision, color=cdict['gray'], alpha=grayalpha, s=30,
           label='1-5px(0.1px刻み)')
for i, annot in zip(range(11), annots):
    
    color = cdict['red']
    if i == 0:
        ax.scatter(df_agt50_avoid5_mean['time'][i],
                   df_agt50_avoid5_mean['collision'][i],
                   color=color, s=size, label='5px', alpha=alpha)
    else:
        ax.scatter(df_agt50_avoid5_mean['time'][i],
                    df_agt50_avoid5_mean['collision'][i],
                    color=color, s=size, alpha=alpha)
    ax.text(df_agt50_avoid5_mean['time'][i]+0.05,
            df_agt50_avoid5_mean['collision'][i]+0.05, 
            annot, size='small')
ax.grid()
ax.legend(title='回避ベクトルの大きさ')

# %%
fig, ax = plt.subplots(figsize=figsize)
plot_continuous(df_agt100_avoid1_mean, color='blue', label='1px')
plot_continuous(df_agt100_avoid2_mean, color='red', label='2px')
plot_continuous(df_agt100_avoid3_mean, color='green', label='3px')
plot_continuous(df_agt100_avoid4_mean, color='orange', label='4px')
plot_continuous(df_agt100_avoid5_mean, color='gray', label='5px')
plt.show()

