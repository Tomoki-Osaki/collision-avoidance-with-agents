"""
メモ (03/17)
Fundamental Diagram (FD, 交通流理論):
    単位時間あたりの車両数である交通量と単位長さあたりの車両数である密度の関係
●密度の増加は走行速度の低下を招く
●ある区間への車両の流入量と流出量が同じであるならば、交通状態は定常である
●臨海密度では、交通状態は不安定である
●臨海密度を超える密度の場合、渋滞が発生する

速度密度関係
速度密度関係は速度と密度には負の相関がある
密度が0に近付くにつれて速度は自由流速度に近付く
密度が増加すると、道路上の車両の速度は低下する
密度が渋滞密度に等しくなると、速度は0になる

流率密度関係
(Greenshields FD)と(三角形FD)
"""
# %% import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation, FuncAnimation
import time 
from gc import collect
from tqdm import tqdm
import os
import pickle
from datetime import datetime, timedelta

import funcSimulation as fs

ped_data = fs.PedData()
ped_data.show_params()

# %% debug simulation
import classSimulation as cs

prepared_data = cs.PreparedData('data_for_awareness_agt25.npz', remove_outliers=2)
#num_agents = 25
#prepared_data = cs.PreparedData.prepare_data(num_agents=num_agents, remove_outliers=2)
#prepared_data = cs.PreparedData('tmp_awareness.npz', remove_outliers=2)
prepared_data.show_params()
#prepared_data.plot_dist('Py')

# %% run simulation
w = cs.AwarenessWeight()
steps = 250

df_fd = pd.DataFrame()

t_now = datetime.now()
print(f'\nシミュレーション開始時刻は {t_now.strftime("%H:%M")} です。\n')

for agent in [100]:
    t = cs.Simulation(random_seed=0,
                      num_steps=steps,
                      num_agents=agent,
                      dynamic_percent=1,
                      simple_avoid_vec=0,
                      viewing_angle=360,
                      prepared_data=prepared_data,
                      awareness_weight=w,
                      awareness=False)
    #t.move_agents()
    
    #t = cs.Simulation(viewing_angle=360, num_steps=100, num_agents=25)
    
    t.simulate()
    
    tmp = t.return_results_as_df()
    print(f'time: {tmp["time"].values[0]:.3f}')
    print(f'collision: {tmp["collision"].values[0]:.3f}')
    print(f'exe time: {tmp["exe_time_min"].values[0]:.2f} min')
    print(f'sum of goals: {tmp["sum_goal_count"].values[0]}')
    
    df_fd = pd.concat([df_fd, tmp])
    df_fd.to_csv(f"processing_agents{agent}.csv", mode='x')
    
    print(f'この試行の終了時刻は {datetime.now().strftime("%H:%M")} です。\n')

#t.save_data_for_awareness(save_as='tmp_awareness.npz')

# %%
df_fd.to_csv('simulation_for_report/increasing_agents_120.csv', mode='x')

# # %%
# with open('agent148_not_averaged.pickle') as f:
#     pickle.dump(t, f)

# %%
path = 'D:/simulation_for_report/'
assert os.path.exists(path), "The folder doesn't exit."

name = f'agent25_avoid0_seed{t.random_seed}.csv'
tmp.to_csv(path+name, mode='x')

df_res = df_fd.reset_index()

# %%
df = pd.DataFrame()
for i in range(0, 5):
    file = f'agent25_avoid0_seed{i}.csv'
    tmpdf = pd.read_csv(path+file)
    df = pd.concat([df, tmpdf])
    
df.to_csv(path+'agent25_avoid0.csv')

# %% make an animation
agt = [i for i in range(t.num_agents)]
agt = 'all'
file = 'tmp.mp4'
fs.animate_agent_movements(sim=t, save_as=file, viz_angle=False, featured_agents=agt)

# %% check how the awareness model was calculated
for i in range(25):
    print(i, t.awareness_model(11, i, w, debug=True))
