"""
メモ (02/19)
動的回避ベクトルを生成する際、近くと遠くの両方にエージェントがいると、遠くのエージェントに対して生成される回避ベクトルが小さいため、
平均されると回避ベクトルが小さくなり、ほとんど回避できないという事態が生じる

awareness_modelにおいて、ループ中に値が全て同じになるバグあり
find_agents_to_focus_with_awareness単体で使用した場合は正常に動くが、move_agents中に呼び出された場合、
全て値が0になる

"""
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation, FuncAnimation
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
prepared_data = cs.PreparedData('data_for_awarenss_agt50.npz', 2)
prepared_data.show_params()
#prepared_data.plot_dist('all_deltaTTCP')

steps = 500
num_agents = 50

t = cs.Simulation(random_seed=0, 
                  num_steps=steps, 
                  num_agents=num_agents, 
                  dynamic_percent=1,
                  prepared_data=prepared_data, 
                  awareness=0.95)
#t.move_agents()

t.simulate()
t.animate_agent_movements(save_as='tmp50.mp4')
res50 = t.return_results_as_df()

# %% animate movements
#w.show_params()

num = 10

fig, ax = plt.subplots(figsize=(8, 8))
ax.grid()
ax.set_xlim(0, 500)
ax.set_ylim(0, 500)
ax.set_xticks(range(0, 501, 50))
ax.set_yticks(range(0, 501, 50))
ax.set_xlabel('Pixel')
ax.set_ylabel('Pixel')

frames = []
for step in tqdm(range(500)):
    artists = []
    #artists.append(ax.set_title(f'{step}'))    
    for i in range(t.num_agents):
        color = 'green' if i == num else 'red'
        
        artists.append(ax.scatter(*(t.all_agents[i]['all_pos'][step]*50)+250,
                                  color=color, alpha=0.6))
        artists.append(ax.text(x=(t.all_agents[i]['all_pos'][step][0]*50)+250, 
                               y=(t.all_agents[i]['all_pos'][step][1]*50)+250, 
                               s=i, size=10))
        if not step == 0:
            tminus1_pos = (t.all_agents[i]['all_pos'][step-1]*50)+250
            artists.append(ax.scatter(*tminus1_pos, color=color, alpha=0.2))
    
    awms = []
    for i in range(t.num_agents):
        awm = t.all_agents[num]['awareness'][step][i]
        if awm >= t.awareness:
            awms.append([i, awm])
                         
    my_posx = (t.all_agents[num]['all_pos'][step][0]*50)+250
    my_posy = (t.all_agents[num]['all_pos'][step][1]*50)+250
    
    for i in awms:
        other_posx = (t.all_agents[i[0]]['all_pos'][step][0]*50)+250
        other_posy = (t.all_agents[i[0]]['all_pos'][step][1]*50)+250
        
        artists.append(ax.arrow(x=my_posx, y=my_posy,
                                dx=other_posx-my_posx, dy=other_posy-my_posy,
                                color='tab:blue', alpha=0.5))
        artists.append(ax.text(x=other_posx-15, y=other_posy-15, 
                               s=np.round(i[1], 2), size=10, color='blue'))
    
    frames.append(artists)

anim = ArtistAnimation(fig, frames, interval=150)
print('drawing the animation...')
anim.save('tmp50_10.mp4')
plt.close() 
