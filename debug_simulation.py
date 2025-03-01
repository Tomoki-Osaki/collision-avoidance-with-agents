"""
メモ (02/21)
動的回避ベクトルを生成する際、近くと遠くの両方にエージェントがいると、遠くのエージェントに対して生成される回避ベクトルが小さいため、
平均されると回避ベクトルが小さくなり、ほとんど回避できないという事態が生じる
ΔTTCPの計算について、エージェントの大きさを考慮し、awarenessの計算時は、ΔTTCPがnanの相手はawarenessを0にする方が良いかも
また、Θの計算について、現状Pyが負の相手のデータを計算していないため、Θについても90度以上はnanとして処理していないため、
ロジスティクス回帰計算時のΘの項の値が大きくなりすぎている(重みも-2.5と大きい)
もし90度以上のΘの値も使うなら、ロジスティクス回帰計算時のPyも負の値を認める必要がある
人としての想定なら自分の後ろ側の歩行者のデータは用いないが、360度センサ付きのロボットの想定では、後ろ側のデータも使える
awareness_modelの引数debug=Trueにすることでロジスティクス回帰計算時の項の値が確認できる
ケース事例を検討する
"""
# %% import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation, FuncAnimation
from matplotlib.patches import Circle
from dataclasses import dataclass
from funcSimulation import show
import funcSimulation as fs
import time 
from gc import collect
from tqdm import tqdm

ped_data = fs.PedData()
ped_data.show_params()

# %% debug simulation
import classSimulation as cs
num_agents = 20

prepared_data = cs.PreparedData('data_for_awareness_agt25.npz', remove_outliers=2)
#prepared_data = cs.PreparedData.prepare_data(num_agents=num_agents, seed=1, remove_outliers=2,
#                                             save_file_as='data_for_awareness_agt25.npz')
#prepared_data = cs.PreparedData('tmp_awareness.npz', remove_outliers=2)
prepared_data.show_params()
#prepared_data.plot_dist('Py')

# %% run simulation
w = cs.AwarenessWeight()
steps = 100

t = cs.Simulation(random_seed=3, 
                  num_steps=steps, 
                  num_agents=num_agents, 
                  dynamic_percent=1,
                  viewing_angle=180,
                  prepared_data=prepared_data,
                  awareness_weight=w,
                  awareness=0.9)
#t.move_agents()

#t = cs.Simulation(viewing_angle=360, num_steps=100, num_agents=25)

t.simulate()

tmp = t.return_results_as_df()
print(f'\n\ntime: {tmp["time"].values[0]:.3f}')
print(f'collision: {tmp["collision"].values[0]:.3f}')

#t.save_data_for_awareness(save_as='tmp_awareness.npz')

# %% make an animation
fs.animate_agent_movements(sim=t, save_as='tmp.mp4', viz_angle=True)

# %% check how the awareness model was calculated
for i in range(25):
    print(i, t.awareness_model(11, i, w, debug=True))
