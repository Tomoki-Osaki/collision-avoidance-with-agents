"""
README (02/10)
スケールを合わせるためにAwareness modelの標準化していない重み係数を教えてもらう必要ありかも？
deltaTTCPの合計は0になるため、平均が0になる
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import classSimulation as cs
import funcSimulation as fs
from funcSimulation import show, standardize
from dataclasses import dataclass

steps = 60
num_agents = 25

t = cs.Simulation(num_steps=steps, num_agents=num_agents)
t.simulate()

@dataclass
class Awareness:
    deltaTTCP: float
    Px: float
    Py: float
    Vself: float
    Vother: float 
    theta: float
    Nic: int
        
all_deltaTTCP = np.array([t.all_agents[i]['deltaTTCP'] for i in range(t.num_agents)])
all_Px = np.array([t.all_agents[i]['relPx'] for i in range(t.num_agents)])
all_Py = np.array([t.all_agents[i]['relPy'] for i in range(t.num_agents)])
all_Vself = np.array([t.all_agents[i]['all_vel'] for i in range(t.num_agents)])
all_Vother = np.array([t.all_agents[i]['all_other_vel'] for i in range(t.num_agents)])
all_theta = np.array([t.all_agents[i]['theta'] for i in range(t.num_agents)])
all_Nic = np.array([t.all_agents[i]['Nic'] for i in range(t.num_agents)])

all_aware = Awareness(all_deltaTTCP, all_Px, all_Py, all_Vself, all_Vother, all_theta, all_Nic)


# %%
def awareness_model(deltaTTCP: float, Px: float, Py: float, 
                    Vself: float, Vother: float, theta: float, Nic: int) -> float:
    """
    Inputsの値は標準化されている必要あり
    
    Inputs
        deltaTTCP: 自分のTTCPから相手のTTCPを引いた値
        Px: 自分から見た相手の相対位置
        Py: 自分から見た相手の相対位置
        Vself: 自分の歩行速度
        Vother: 相手の歩行速度 
        theta: 自分の向いている方向と相手の位置の角度差 
        Nic: 円内他歩行者数 
    Output
        0(注視しない) - 1 (注視する)
    """
    if deltaTTCP is None:
        return 0
    
    deno = 1 + np.exp(
        -(-1.2 +0.018*deltaTTCP -0.1*Px -1.1*Py \
          -0.25*Vself +0.29*Vother -2.5*theta -0.62*Nic)    
    )
    val = 1 / deno
    
    return val
