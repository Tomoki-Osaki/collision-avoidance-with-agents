"""
メモ (02/12)
スケールを合わせるためにAwareness modelの標準化していない重み係数を教えてもらう必要ありかも？
deltaTTCPの合計は0になるため、平均が0になる
現状のPxとPyは絶対座標をもとにしているが、エージェントの向きをy座標とした座標系を構築する必要あり
また、このときPxは絶対値を取るようにする
VselfとVotherはノルムを計算しているため、新しく座標系を構築する必要はないと思われる
classSimulationの関数を一部funcSimulationに移動させる
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from dataclasses import dataclass
from funcSimulation import show, standardize
import funcSimulation as fs

ped_data = fs.PedData()
ped_data.show_params()

prepared_data = np.load('tmp_comp.npz')
print(prepared_data['all_Px'].shape) # agent, steps, other_agent
show(prepared_data.files)

def stats(key):
    print('mean', np.nanmean(prepared_data[key]))
    print('std', np.nanstd(prepared_data[key]))

def hist(key, bins=100):
    x = np.where(~np.isnan(prepared_data[key]))[0]
    plt.hist(x, bins=bins)

arr1 = np.array([1, 2])
vel1 = np.array([0.5, 0.2])

arr2 = np.array([6, 9])
vel2 = np.array([-1, -3])

fs.calc_deltaTTCP(arr1, vel1, arr2, vel2)

import classSimulation as cs
steps = 100
num_agents = 25

t = cs.Simulation(random_seed=10, num_steps=steps, num_agents=num_agents, dynamic_percent=.5)
t.simulate()

#t.save_data_for_awareness(save_as='tmp_comp.npz')

step = 45
t.plot_positions(step)

# agent22 to agent8 step45
# w = AwarenessWeight(deltaTTCP=20, Nic=-4, theta=-0.5)
# t.plot_positions_aware(agent=22, step=45, all_aware, w)        
w = fs.AwarenessWeight.multiple(5)
w.show_params()

agent = 22
t.plot_positions_aware(agent, step, prepared_data, w)

for i in range(25):
    print(i, fs.awareness_model(t, agent, i, step, prepared_data, w, debug=True))
    
    
for step in range(35, 45):
    t.plot_positions_aware(agent, step, prepared_data, w)

