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
from funcSimulation import show, standardize
import funcSimulation as fs
import time 

prepared_data = np.load('data_for_awarenss_agt25.npz')
print('shape:', prepared_data['all_Px'].shape) # agent, steps, other_agent
show(prepared_data.files)

# %%
import classSimulation as cs
steps = 200
num_agents = 25

t = cs.Simulation(random_seed=5, 
                  num_steps=steps, 
                  num_agents=num_agents, 
                  dynamic_percent=1,
                  prepared_data=prepared_data, 
                  awareness=True)

start = time.perf_counter()
t.simulate()
print('\n', time.perf_counter() - start)
df_res25 = t.return_results_as_df()
#t.save_data_for_awareness(save_as='tmp_comp.npz')

step = 500
t.plot_positions(step)

# agent22 to agent8 step45
# w = AwarenessWeight(deltaTTCP=20, Nic=-4, theta=-0.5)
# t.plot_positions_aware(agent=22, step=45, all_aware, w)    
w = fs.AwarenessWeight()    
w = fs.AwarenessWeight.multiple(3)
w.show_params()

agent = 28
t.plot_positions_aware(agent, step, prepared_data, w)

for i in range(num_agents):
    print(i, fs.awareness_model(t, agent, i, step, prepared_data, w, debug=True))


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
