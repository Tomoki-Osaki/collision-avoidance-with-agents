"""
メモ (02/11)
スケールを合わせるためにAwareness modelの標準化していない重み係数を教えてもらう必要ありかも？
deltaTTCPの合計は0になるため、平均が0になる
現状のPxとPyは絶対座標をもとにしているが、エージェントの向きをy座標とした座標系を構築する必要あり
また、このときPxは絶対値を取るようにする
VselfとVotherはノルムを計算しているため、新しく座標系を構築する必要はないと思われる
awareness modelではエージェント間の距離が考慮されていないため、非常に離れた相手とも、ΔTTCPが計算される(交点が計算できる)
場合、特にthetaやNicの値によっては注視対象として計算される
注視対象として計算されるだけなら問題ないが、その注視対象が非常に離れているにもかかわらずその対象に対して回避ベクトルを計算するのは不自然である
よって、計算対象の選定の後、実際にその計算対象に対して回避ベクトルを計算するかどうかという決定においてブレーキ指標が使えるのではないか
つまり、計算対象として選定されていて、かつブレーキ指標が一定以上の値を取るエージェントに対して回避ベクトルを生成する
ブレーキ指標は最接近距離と最接近点までの時間というふうに、エージェント同士の距離を使用しているため、エージェント同士が近いときと遠い時で異なる値が産出される
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from funcSimulation import show, standardize
from dataclasses import dataclass
from dataclass_csv import DataclassWriter
import funcSimulation as fs

import classSimulation as cs
steps = 500
num_agents = 25

t = cs.Simulation(random_seed=0, num_steps=steps, num_agents=num_agents, dynamic_percent=.5)
t.simulate()

step = 50
t.plot_positions(step)

agent = 22
for i in range(25):
    print(i, t.awareness_model(agent, i, step, all_aware, debug=True))
    
t.plot_positions_aware(agent, step, all_aware)

# %%
fig, ax = plt.subplots(figsize=(8, 8))
artists = []
agent = 23

for step in range(30, 50):
    txt_far = 0.05
    for i in range(t.num_agents):
        if i == agent:
            color = 'green'
        else:
            color = 'blue'
        scatter = ax.scatter(*t.all_agents[i]['all_pos'][step],
                             color=color, alpha=0.6)
        annot = ax.annotate(i, xy=(t.all_agents[i]['all_pos'][step][0]+txt_far, 
                                   t.all_agents[i]['all_pos'][step][1]+txt_far))
        if not step == 0:
            tminus1_pos = t.all_agents[i]['all_pos'][step-1]
            scatter_minust1 = ax.scatter(*tminus1_pos, color=color, alpha=0.2)
    
    awms = []
    for i in range(t.num_agents):
        awm = t.awareness_model(agent, i, step, all_aware, debug=False)
        if awm is not None and awm >= 0.8:
            awms.append([i, awm])
    
    my_posx = t.all_agents[agent]['all_pos'][step][0]
    my_posy = t.all_agents[agent]['all_pos'][step][1]
    
    for i in awms:
        other_posx = t.all_agents[i[0]]['all_pos'][step][0]
        other_posy = t.all_agents[i[0]]['all_pos'][step][1]
        
        arrow = ax.arrow(x=my_posx, y=my_posy,
                         dx=other_posx-my_posx, dy=other_posy-my_posy,
                         color='tab:blue', alpha=0.5)
    
    grid = ax.grid()   
    artists.append([scatter, scatter_minust1, annot, arrow, grid])
    
anim = ArtistAnimation(fig, artists)
anim.save('tmp.mp4')

# %%
@dataclass
class Awareness:
    deltaTTCP: np.array
    Px: np.array
    Py: np.array
    Vself: np.array
    Vother: np.array 
    theta: np.array
    Nic: np.array
        
all_deltaTTCP = np.array([t.all_agents[i]['deltaTTCP'] for i in range(t.num_agents)])
all_Px = np.array([t.all_agents[i]['relPx'] for i in range(t.num_agents)])
all_Py = np.array([t.all_agents[i]['relPy'] for i in range(t.num_agents)])
all_Vself = np.array([t.all_agents[i]['all_vel'] for i in range(t.num_agents)])
all_Vother = np.array([t.all_agents[i]['all_other_vel'] for i in range(t.num_agents)])
all_theta = np.array([t.all_agents[i]['theta'] for i in range(t.num_agents)])
all_Nic = np.array([t.all_agents[i]['Nic'] for i in range(t.num_agents)])

all_aware = Awareness(all_deltaTTCP, all_Px, all_Py, all_Vself, all_Vother, all_theta, all_Nic)

awares = [all_aware]
with open('all_aware_seed0_steps500.csv', 'w') as f:
    w = DataclassWriter(f, awares, Awareness)
    w.write()
