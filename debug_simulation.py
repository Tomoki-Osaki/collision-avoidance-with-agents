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
#prepared_data = cs.PreparedData('data_for_awarenss_agt25.npz', 2)
prepared_data = cs.PreparedData('tmp_awareness.npz', 2)
#prepared_data = cs.PreparedData.prepare_data(num_agents=25, seed=0, remove_outliers_deltaTTCP=2)
prepared_data.show_params()
prepared_data.plot_dist('all_deltaTTCP')

steps = 500
num_agents = 25

t = cs.Simulation(random_seed=0, 
                  num_steps=steps, 
                  num_agents=num_agents, 
                  dynamic_percent=1,
                  prepared_data=prepared_data, 
                  awareness=0.9)
#t.move_agents()

#t = cs.Simulation(viewing_angle=360, num_steps=100, num_agents=25)

t.simulate()

#t.save_data_for_awareness(save_as='tmp_awareness.npz')

# %%
t.animate_agent_movements(save_as='tmp.mp4')
res25 = t.return_results_as_df()

# %%
w = cs.AwarenessWeight()
for i in range(25):
    print(i, t.awareness_model(11, i, w, debug=True))

# %% adjust the delta TTCP
agent1 = t.all_agents[13]
agent2 = t.all_agents[18]
# 21 14
step = 10
step_tminus1 = step - 1
posx1, posy1 = agent1['all_pos'][step]
posx1_tminus1, posy1_tminus1 = agent1['all_pos'][step_tminus1]
posx2, posy2 = agent2['all_pos'][step]
posx2_tminus1, posy2_tminus1 = agent2['all_pos'][step_tminus1]

# line equation and slope
# y = mx + b
m1 = (posy1 - posy1_tminus1) / (posx1 - posx1_tminus1)
b1 = posy1_tminus1 - m1 * posx1_tminus1

m2 = (posy2 - posy2_tminus1) / (posx2 - posx2_tminus1)
b2 = posy2_tminus1 - m2 * posx2_tminus1

x1 = np.linspace(0, 5)
x2 = np.linspace(-5, 2)

# %% plot
fig, ax = plt.subplots()
ax.scatter(posx1, posy1, c='b')
ax.text(posx1+0.05, posy1, s=f'[{posx1:.1f},{posy1:.1f}]')

ax.scatter(posx2, posy2, c='r')
ax.text(posx2+0.05, posy2, s=f'[{posx2:.1f},{posy2:.1f}]')

ax.scatter(posx1_tminus1, posy1_tminus1, c='b', alpha=.3)
ax.scatter(posx2_tminus1, posy2_tminus1, c='r', alpha=.3)

line1_b = b1 + t.agent_size
line2_b = b1 - t.agent_size
line3_b = b2 + t.agent_size
line4_b = b2 - t.agent_size
line1_m = line2_m = m1
line3_m = line4_m = m2

# y = ax + b
line1 = line1_m * x1 + line1_b
line2 = line2_m * x1 + line2_b
line3 = line3_m * x2 + line3_b
line4 = line4_m * x2 + line4_b

x13 = -( (line1_b - line3_b) / (line1_m - line3_m) )
y13 = line1_m * x13 + line1_b

x14 = -( (line1_b - line4_b) / (line1_m - line4_m) )
y14 = line1_m * x14 + line1_b

x23 = -( (line2_b - line3_b) / (line2_m - line3_m) )
y23 = line2_m * x23 + line2_b

x24 = -( (line2_b - line4_b) / (line2_m - line4_m) )
y24 = line2_m * x24 + line2_b

cps = [np.array([x13, y13]), np.array([x14, y14]), 
       np.array([x23, y23]), np.array([x24, y24])]

agent1_pos = np.array([posx1, posy1])
agent2_pos = np.array([posx2, posy2])

min_dist = None
cp = None
for value in cps:
    agent1_to_cp = np.linalg.norm(agent1_pos - value)
    agent2_to_cp = np.linalg.norm(agent2_pos - value)
    sum_dist = agent1_to_cp + agent2_to_cp

    if not min_dist or sum_dist < min_dist:
        min_dist = sum_dist
        cp = value

    
ax.set_xlim(-6, 6)
ax.set_ylim(-20, 2)

#ax.plot(x1, m1*x1+b1, c='b', alpha=0.3)
ax.plot(x1, m1*x1+(b1+0.1), c='b', alpha=0.6, label='line1')
ax.plot(x1, m1*x1+(b1-0.1), c='indigo', alpha=0.3, label='line2')

#ax.plot(x2, m2*x2+b2, c='r', alpha=0.3)
ax.plot(x2, m2*x2+(b2+0.1), c='r', alpha=0.6, label='line3')
ax.plot(x2, m2*x2+(b2-0.1), c='brown', alpha=0.3, label='line4')

ax.scatter(x13, y13, label='cp13')
ax.scatter(x14, y14, label='cp14')
ax.scatter(x23, y23, label='cp23')
ax.scatter(x24, y24, label='cp24')

ax.grid()
ax.legend()
plt.show()


# %% animate movements
#w.show_params()
num = 8

fig, ax = plt.subplots(figsize=(8, 8))
ax.grid()
ax.set_xlim(0, 500)
ax.set_ylim(0, 500)
ax.set_xticks(range(0, 501, 50))
ax.set_yticks(range(0, 501, 50))
ax.set_xlabel('Pixel')
ax.set_ylabel('Pixel')

frames = []
for step in tqdm(range(200)):
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
anim.save('tmp25_10.mp4')
plt.close() 
