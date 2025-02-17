"""
メモ (02/12)
スケールを合わせるためにAwareness modelの標準化していない重み係数を教えてもらう必要ありかも？
deltaTTCPの合計は0になるため、平均が0になる
現状のPxとPyは絶対座標をもとにしているが、エージェントの向きをy座標とした座標系を構築する必要あり
また、このときPxは絶対値を取るようにする
VselfとVotherはノルムを計算しているため、新しく座標系を構築する必要はないと思われる
classSimulationの関数を一部funcSimulationに移動させる
"""
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from dataclasses import dataclass
from funcSimulation import show
import funcSimulation as fs
import time 
from gc import collect

prepared_data = fs.PreparedData('data_for_awarenss_agt25.npz')
prepared_data.show_params()

# %%
import classSimulation as cs
steps = 300
num_agents = 25

t = cs.Simulation(random_seed=0, 
                  num_steps=steps, 
                  num_agents=num_agents, 
                  dynamic_percent=1,
                  prepared_data=prepared_data, 
                  awareness=True)

t.simulate()

t.move_agents()
aw = t.find_agents_to_focus_with_awareness(0, 0)
dist_all = t.calc_distance_all_agents()
vis = t.find_visible_agents(dist_all, 0)
for i in range(num_agents):
    print(t.dynamic_avoidance(i))

#df_res25 = t.return_results_as_df()
#t.save_data_for_awareness(save_as='data_for_awarenss_agt25.npz')

t.animate_agent_movements(save_as='awareness.mp4')

step = 500
t.plot_positions(step)

w = cs.AwarenessWeight()    
w.show_params()

agent = 7
t.plot_positions_aware(agent, 1, prepared_data, w)

for i in range(num_agents):
    print('agent', i)
    print(t.awareness_model(agent, i, prepared_data, w, debug=True))


# %%
ped_data = fs.PedData()
ped_data.show_params()

# key = [all_deltaTTCP, all_Px, all_Py, all_Vself, all_Vother, all_theta, all_Nic]
def stats(key):
    print('mean', np.nanmean(prepared_data[key]))
    print('std', np.nanstd(prepared_data[key]))

def hist(key, bins=20):
    x = np.where(~np.isnan(prepared_data[key]))[0]
    plt.hist(x, bins=bins)

pos1 = t.all_agents[agent]['p']
vel1 = t.all_agents[agent]['v']

pos1 = t.all_agents[agent]['p']
vel1 = t.all_agents[agent]['v']
