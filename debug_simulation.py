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
from funcSimulation import show, standardize
from dataclasses import dataclass
import funcSimulation as fs

# awareness modelの重み
@dataclass
class AwarenessWeight:
    """
    値は標準化されている必要あり
    """
    bias: float = -1.2
    deltaTTCP: float = 0.018
    Px: float = -0.1
    Py: float = -1.1
    Vself: float = -0.25
    Vother: float = 0.29
    theta: float = -2.5
    Nic: float = -0.62

    @staticmethod
    def multiple(k):
        return AwarenessWeight(
            bias = -1.2 * k,
            deltaTTCP = 0.018 * k,
            Px = -0.1 * k,
            Py = -1.1 * k,
            Vself = -0.25 * k,
            Vother = 0.29 * k,
            theta = -2.5 * k,
            Nic = -0.62 * k
        )


import classSimulation as cs
steps = 500
num_agents = 25

t = cs.Simulation(random_seed=10, num_steps=steps, num_agents=num_agents, dynamic_percent=.5)
t.simulate()

t.save_data_for_awareness(save_as='tmp_comp.npz')

step = 45
t.plot_positions(step)

# agent22 to agent8 step45
# w = AwarenessWeight(deltaTTCP=20, Nic=-4, theta=-0.5)
# t.plot_positions_aware(agent=22, step=45, all_aware, w)        
w = AwarenessWeight.multiple(1)

agent = 22
for i in range(25):
    print(i, fs.awareness_model(t, agent, i, step, all_aware, w, debug=True))
    
for step in range(35, 45):
    t.plot_positions_aware(agent, step, all_aware, w)
