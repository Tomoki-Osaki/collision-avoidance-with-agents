"""
メモ (02/20)
動的回避ベクトルを生成する際、近くと遠くの両方にエージェントがいると、遠くのエージェントに対して生成される回避ベクトルが小さいため、
平均されると回避ベクトルが小さくなり、ほとんど回避できないという事態が生じる
ΔTTCPの計算について、エージェントの大きさを考慮し、awarenessの計算時は、ΔTTCPがnanの相手はawarenessを0にする方が良いかも
また、Θの計算について、現状Pyが負の相手のデータを計算していないため、Θについても90度以上はnanとして処理していないため、
ロジスティクス回帰計算時のΘの項の値が大きくなりすぎている(重みも-2.5と大きい)
もし90度以上のΘの値も使うなら、ロジスティクス回帰計算時のPyも負の値を認める必要がある
人としての想定なら自分の後ろ側の歩行者のデータは用いないが、360度センサ付きのロボットの想定では、後ろ側のデータも使える
awareness_modelの引数debug=Trueにすることでロジスティクス回帰計算時の項の値が確認できる
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
prepared_data = cs.PreparedData('data_for_awarenss_agt25.npz', remove_outliers=2)
prepared_data = cs.PreparedData('tmp_awareness.npz', remove_outliers=2)
prepared_data = cs.PreparedData.prepare_data(num_agents=50, seed=1, remove_outliers=2,
                                             save_file_as='data_for_awareness_agt50.npz')
prepared_data.show_params()
prepared_data.plot_dist('Py')

# %%
w = cs.AwarenessWeight()
steps = 500
num_agents = 50

t = cs.Simulation(random_seed=0, 
                  num_steps=steps, 
                  num_agents=num_agents, 
                  dynamic_percent=1,
                  prepared_data=prepared_data,
                  awareness_weight=w,
                  awareness=0.9)
#t.move_agents()

#t = cs.Simulation(viewing_angle=360, num_steps=100, num_agents=25)

t.simulate()

#t.save_data_for_awareness(save_as='tmp_awareness.npz')

# %%
t.animate_agent_movements(save_as='tmp.mp4')
res50 = t.return_results_as_df()

# %%
for i in range(25):
    print(i, t.awareness_model(11, i, w, debug=True))

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
s = 45
frames = []
for step in tqdm(range(500)):
    artists = []
    #artists.append(ax.set_title(f'{step}'))    
    for i in range(t.num_agents):
        color = 'green' if i == num else 'red'
        
        artists.append(ax.scatter(*(t.all_agents[i]['all_pos'][step]*50)+250,
                                  color=color, alpha=0.6, s=s))
        artists.append(ax.text(x=(t.all_agents[i]['all_pos'][step][0]*50)+250, 
                               y=(t.all_agents[i]['all_pos'][step][1]*50)+250, 
                               s=i, size=10))
        if not step == 0:
            tminus1_pos = (t.all_agents[i]['all_pos'][step-1]*50)+250
            artists.append(ax.scatter(*tminus1_pos, color=color, alpha=0.2, s=s))
    
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
print('\ndrawing the animation...')
anim.save('tmp_awareness.mp4')
plt.close() 
