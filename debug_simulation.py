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

prepared_data = np.load('tmp_comp.npz')
print('shape:', prepared_data['all_Px'].shape) # agent, steps, other_agent
show(prepared_data.files)

# %%
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


# %%
ped_data = fs.PedData()
ped_data.show_params()

def stats(key):
    print('mean', np.nanmean(prepared_data[key]))
    print('std', np.nanstd(prepared_data[key]))

def hist(key, bins=100):
    x = np.where(~np.isnan(prepared_data[key]))[0]
    plt.hist(x, bins=bins)


# =============================================================================
# relPxとrelPyを求めるために、
# [1]. Pt-1からPtの方向に延長線を引く
# [2]. [1]と垂直でPtを通る直線を引く
# [3]. [1]に垂直で相手のPを通る直線を引き、交点を求める
# [4]. 相手のPと[3]の交点までの距離を求め、これがrelPxになる
# [5]. [2]に垂直で相手のPを通る直線を引き、交点を求める
# [6]. 相手のPのy座標から[5]の交点のy座標を引き、これがrelPyになる(relPxと違い、絶対値は求めない)
# =============================================================================

arr1 = np.array([1, 2])
vel1 = np.array([0.5, 0.2])
arr1_next = arr1 + vel1 # array([1.5, 2.2])
#延長線は線分である必要はない
ext_line1 = fs.extend_line(arr1, arr1_next, 5)

arr2 = np.array([4, 5])
vel2 = np.array([-0.5, -0.8])
arr2_next = arr2 + vel2 # array([3.5, 4.2])
ext_line2 = fs.extend_line(arr2, arr2_next, 5)

# 傾きと切片を求める
# y = ax + b
# b = y - ax
# 2直線が垂直 → a*a' = -1 (a' = -(1 / a))
posx1_tminus1, posy1_tminus1 = arr1
posx1, posy1 = arr1_next
posx2, posy2 = arr2_next

# (1) agent1のPt-1からPtに伸びる直線の傾きと切片
a1 = (posy1 - posy1_tminus1) / (posx1 - posx1_tminus1)
b1 = posy1_tminus1 - a1 * posx1_tminus1

# (2) agent1のPt-1からPtに伸びる直線に対して垂直な直線の傾きと切片
v_a1 = -(1 / a1)
v_b1 = posy1 - v_a1 * posx1

# (3) agent2のPtから(2)に対して垂直な直線の傾きと切片
a2 = v_a1
b2 = posy2 - a2 * posx2

# (4) (1)と(3)の交点
x = (b1 - b2) / (a2 - a1) 
y = a2 * posx2 + b2



xs1 = np.linspace(posx1, ext_line1[0])
ys1 = np.linspace(posy1, ext_line1[1])
xs2 = np.linspace(posx2, ext_line2[0])
ys2 = np.linspace(posy2, ext_line2[1])
plt.plot(xs1, ys1, c='b')
plt.plot(xs2, ys2, c='r')
plt.scatter(x, y, c='g')
plt.grid()



fs.calc_deltaTTCP(arr1, vel1, arr2, vel2)

plt.scatter(*arr1, c='b', alpha=.3)
plt.scatter(*arr1_next, c='b')
plt.scatter(*arr2_next, c='r')
plt.scatter(x, y, c='g')
plt.grid()

plt.plot(*zip(arr1, arr1_next))



