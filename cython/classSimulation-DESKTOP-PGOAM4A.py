""" 
シミュレーションのためのクラスSimulation

class Simulation
__init__(self)
1. set_goals(agent)
2. calc_distance_all_agents()
3. find_visible_agents(num)
4. simple_avoidance(num)
5. dynamic_avoidance(num)
6. calc_Nic(num)
7. find_agents_to_focus(num)
8. record_start_and_goal(num)
9. calc_completion_time(num)
10. calc_last_completion_time(num)
11. check_if_goaled()
12. simulate()
13. show_image()
14. plot_positions()
15. approach_detect(dist)
16. record_agent_information()
17. record_approaches(approach_dist, step, sim_data)
"""

# %% import libraries
from copy import deepcopy
from typing import Literal
import time 
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import funcSimulation as fs

# %% 
# グラフの目盛りの最大値・最小値
FIELD_SIZE = 5 
# 目盛りは最大値5、最小値-5で10目盛り
# グラフ領域の幅と高さは500pxなので、1pxあたり0.02目盛りとなる

# %% クラス： BrakeWeight, AwarenessWeight, PreparedData
@dataclass
class BrakeWeight():
    """
    ブレーキ指標の4係数は標準化したものを使用する
    """
    a1: float = -5.145 # -0.034298
    b1: float = 3.348 # 3.348394
    c1: float = 4.286 # 4.252840
    d1: float = -13.689 # -0.003423
    
    
@dataclass
class AwarenessWeight():
    """
    awareness modelの重み
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

    def show_params(self) -> None:
        print('\nWeights of Awareness model')
        print('-------------------')
        print('bias:', np.round(self.bias, 4))
        print('deltaTTCP:', np.round(self.deltaTTCP, 4))
        print('Px:', np.round(self.Px, 4))
        print('Py:', np.round(self.Py, 4))
        print('Vself:', np.round(self.Vself, 4))
        print('Vother:', np.round(self.Vother, 4))
        print('theta:', np.round(self.theta, 4))
        print('Nic:', np.round(self.Nic, 4))
        print('-------------------\n')
        
        
class PreparedData:
    """
    Awareness modelをシミュレーションで適用する際、事前にシミュレーションを行い各説明変数の平均と標準偏差を
    求める必要がある
    その際の平均と標準偏差を格納するクラス
    """
    def __init__(self, npz_file_path: str, remove_outliers=None):
        self.data = np.load(npz_file_path)
        
        self.deltaTTCP = np.ravel(self.data['all_deltaTTCP'])
        self.Px = np.ravel(self.data['all_Px'])
        self.Py = np.ravel(self.data['all_Py'])
        self.Vself = np.ravel(self.data['all_Vself'])
        self.Vother = np.ravel(self.data['all_Vother'])
        self.theta = np.ravel(self.data['all_theta'])
        self.Nic = np.ravel(self.data['all_Nic'])
                
        self.deltaTTCP = fs.remove_outliers_and_nan(self.deltaTTCP, sd=remove_outliers)
        self.Px = fs.remove_outliers_and_nan(self.Px, sd=remove_outliers)
        self.Py = fs.remove_outliers_and_nan(self.Py, sd=remove_outliers)
        self.Vself = fs.remove_outliers_and_nan(self.Vself, sd=remove_outliers)
        self.Vother = fs.remove_outliers_and_nan(self.Vother, sd=remove_outliers)
        self.theta = fs.remove_outliers_and_nan(self.theta, sd=remove_outliers)
        self.Nic = fs.remove_outliers_and_nan(self.Nic, sd=remove_outliers)
            
        self.deltaTTCP_mean, self.deltaTTCP_std = np.mean(self.deltaTTCP), np.std(self.deltaTTCP)
        
        self.Px_mean, self.Px_std = np.mean(self.Px), np.std(self.Px)
        
        self.Py_mean, self.Py_std = np.mean(self.Py), np.std(self.Py)
        
        self.Vself_mean, self.Vself_std = np.mean(self.Vself), np.std(self.Vself)
        
        self.Vother_mean, self.Vother_std = np.mean(self.Vother), np.std(self.Vother)
        
        self.theta_mean, self.theta_std = np.mean(self.theta), np.std(self.theta)
        
        self.Nic_mean, self.Nic_std = np.mean(self.Nic), np.std(self.Nic)
    
    
    def show_files(self) -> None:
        for i, j in enumerate(self.data.files):
            print(i, j)
    
    
    def show_params(self) -> None:
        print('\nParameters of pre-run simulation')
        print('-------------------------')
        print('deltaTTCP_mean:', np.round(self.deltaTTCP_mean, 3))
        print('deltaTTCP_std:', np.round(self.deltaTTCP_std, 3))
        print('mean-std ratio:', np.round(self.deltaTTCP_mean/self.deltaTTCP_std, 3))
        
        print('\nPx_mean:', np.round(self.Px_mean, 3)) 
        print('Px_std:', np.round(self.Px_std, 3))
        print('mean-std ratio:', np.round(self.Px_mean/self.Px_std, 3))        
        
        print('\nPy_mean:', np.round(self.Py_mean, 3))
        print('Py_std:', np.round(self.Py_std, 3))
        print('mean-std ratio:', np.round(self.Py_mean/self.Py_std, 3))
        
        print('\nVself_mean:', np.round(self.Vself_mean, 3))
        print('Vself_std:', np.round(self.Vself_std, 3))
        print('mean-std ratio:', np.round(self.Vself_mean/self.Vself_std, 3))
        
        print('\nVother_mean:', np.round(self.Vother_mean, 3))
        print('Vother_std:', np.round(self.Vother_std, 3))
        print('mean-std ratio:', np.round(self.Vother_mean/self.Vother_std, 3))
        
        print('\ntheta_mean:', np.round(self.theta_mean, 3))
        print('theta_std:', np.round(self.theta_std, 3))
        print('mean-std ratio:', np.round(self.theta_mean/self.theta_std, 3))
        
        print('\nNic_mean:', np.round(self.Nic_mean, 3))
        print('Nic_std:', np.round(self.Nic_std, 3))
        print('mean-std ratio:', np.round(self.Nic_mean/self.Nic_std, 3))
        print('-------------------------\n')
    
    
    def plot_dist(self, 
                  key: Literal['deltaTTCP', 'Px', 'Py', 'Vself', 
                               'Vother', 'theta', 'Nic']) -> None:
        if key == 'deltaTTCP': x = self.deltaTTCP
        elif key == 'Px': x = self.Px
        elif key == 'Py': x = self.Py    
        elif key == 'Vself': x = self.Vself
        elif key == 'Vother': x = self.Vother
        elif key == 'theta': x = self.theta
        elif key == 'Nic': x = self.Nic

        fig, ax = plt.subplots()
        ax.violinplot(x, side='high', vert=False)
        ax.boxplot(x, vert=False)
        ax.set_title(key)
        ax.grid()
        plt.show()
    
    
    @staticmethod
    def prepare_data(num_agents, remove_outliers=None, save_file_as='tmp.npz'):
        sim = Simulation(num_agents=num_agents, random_seed=100)
        print('\nPreparing data for awareness model...')
        sim.simulate()
        sim.save_data_for_awareness(save_as=save_file_as)
        print(f'\ntemporary data files is [{save_file_as}]')
        prepared_data = PreparedData(save_file_as, remove_outliers)
        
        return prepared_data
        
    
# %% シミュレーションに関わるクラス
class Simulation:
    def __init__(self, 
                 num_steps: int = 500,
                 interval: int = 100,
                 agent_size: float = 0.1, 
                 num_agents: int = 25, 
                 view: int = 1, 
                 viewing_angle: int = 360, 
                 goal_vec: float = 0.06, 
                 dynamic_percent: float = 1.0,
                 simple_avoid_vec: float = 0.06, 
                 dynamic_avoid_vec: float = 0.06,
                 prepared_data: PreparedData = None,
                 awareness_weight: AwarenessWeight = None,
                 awareness: bool or float = False,
                 random_seed: int = 0):
        
        if awareness:
            assert prepared_data, 'Awareness model needs the prepared data'
            
        self.num_steps = num_steps
        self.interval = interval # 100msごとにグラフを更新してアニメーションを作成
        self.agent_size = agent_size # エージェントの半径(目盛り) = 5px
        self.num_agents = num_agents # エージェント数
        self.view = view # 視野の半径(目盛り) = 50px:エージェント5体分
        self.viewing_angle = viewing_angle # 視野の角度
        self.goal_vec = goal_vec # ゴールベクトルの大きさ(目盛り)
        self.dynamic_percent = dynamic_percent # 動的回避を基に回避するエージェントの割合
        self.simple_avoid_vec = simple_avoid_vec # 単純回避での回避ベクトルの大きさ(目盛り)
        self.dynamic_avoid_vec = dynamic_avoid_vec # 動的回避での回避ベクトルの最大値(目盛り)
        self.random_seed = random_seed
        self.prepared_data = prepared_data
        self.awareness_weight = awareness_weight
        self.awareness = awareness
        
        self.all_agents = [] # 全エージェントの座標を記録
        self.all_agents2 = [] # ゴールの計算用
        self.first_agent = [] # 初期位置記録用
        self.agent_goals = []
        self.first_pos =[]
        
        #エージェント間の距離を記録するリスト
        self.dist = np.zeros([self.num_agents, self.num_agents])
                
        # 動的回避を行うエージェントの数
        self.num_dynamic_agent = int(np.round(self.num_agents*self.dynamic_percent))
        
        # エージェントの生成
        np.random.seed(self.random_seed)
        for n in range(self.num_agents):
            # グラフ領域の中からランダムに座標を決定
            pos = np.random.uniform(-FIELD_SIZE, FIELD_SIZE, 2)
            vel = np.random.uniform(-FIELD_SIZE, FIELD_SIZE, 2)
            
            # 1は動的回避で、0は単純回避
            avoidance = 1 if n < self.num_dynamic_agent else 0
            
            # 座標(0, 0)から座標velへのベクトルがエージェントの初期速度になる
            # self.all_agentsの1つの要素に1体のエージェントの位置と速度が記録
            # P(t) - V(t) = P(t-1)
            self.all_agents.append(
                {
                    'avoidance': avoidance, 
                    'p': pos, 
                    'v': fs.rotate_vec(np.array([self.goal_vec, 0]), 
                                       fs.calc_rad(vel, np.array([0, 0]))),
                    'all_pos': np.zeros([self.num_steps+1, 2]),
                    'all_vel': np.zeros(self.num_steps+1),
                    # # ↓他のそれぞれのエージェントに対して算出されるパラメータ
                    # 'all_other_vel': np.zeros([self.num_steps+1, self.num_agents]),
                    # 'relPx': np.zeros([self.num_steps+1, self.num_agents]),
                    # 'relPy': np.zeros([self.num_steps+1, self.num_agents]),
                    # 'theta': np.zeros([self.num_steps+1, self.num_agents]),
                    # 'deltaTTCP': np.zeros([self.num_steps+1, self.num_agents]),
                    # 'Nic': np.zeros([self.num_steps+1, self.num_agents]),
                    # 'awareness': np.zeros([self.num_steps+1, self.num_agents])
                }
            )
            
        # 初期位置と初期速度をコピー
        self.all_agents2 = deepcopy(self.all_agents)
        self.first_agent = deepcopy(self.all_agents)
        
        # エージェントの初期位置を保存
        for i in range(self.num_agents):
            self.first_pos.append(self.first_agent[i]['p'])
        
        # エージェントにゴールを8ずつ設定
        for i in range(self.num_agents):
            goals = [self.set_goals(self.all_agents2[i]) for _ in range(8)]
            self.agent_goals.append(goals)
            
        # ゴールした回数を記録するリスト
        self.goal_count = [0] * self.num_agents
        
        # はみ出た時用のゴール
        self.goal_temp = np.zeros([self.num_agents, 2])
        
        # 完了時間を測るための変数
        self.start_step = np.zeros([self.num_agents])
        self.goal_step = np.zeros([self.num_agents])
        self.start_pos = self.first_pos
        self.goal_pos = np.zeros([self.num_agents, 2])
         
        # 完了時間を記録するリスト
        self.completion_time = []
        
        self.data = []
        row = self.record_agent_information() # 全エージェントの位置と速度、接近を記録
        self.data.append(row) # ある時刻でのエージェントの情報が記録されたrowが集まってdataとなる
        
        self.exe_time = None
        
        self.current_step = 0
        self.update_parameters()
        
        self.returned_results = False
    
    
    def disp_info(self) -> None:
        print('\nシミュレーションの環境情報')
        print('-----------------------------')
        print('ランダムシード:', self.random_seed)
        print('ステップ数:', self.num_steps)
        print('エージェント数:', self.num_agents)
        print('エージェントの視野角度:', self.viewing_angle)
        print('動的回避エージェントの割合:', self.dynamic_percent)
        print('単純回避の回避量:', self.simple_avoid_vec*50, 'px')
        print('Awareness:', self.awareness)
        print('-----------------------------\n')
    
    
    # 1
    def update_parameters(self) -> None:
        """
        all_agentsのパラメータを更新する
        """
        # all_vel, all_pos
        for i in range(self.num_agents):
            pos = self.all_agents[i]['p']
            vel = np.linalg.norm(self.all_agents[i]['v'])

            self.all_agents[i]['all_pos'][self.current_step][0] = pos[0] # x
            self.all_agents[i]['all_pos'][self.current_step][1] = pos[1] # y           
            self.all_agents[i]['all_vel'][self.current_step] = vel
        
        ## 自分と相手の情報を基に計算するパラメータ
        # for i in range(self.num_agents):
        #     for j in range(self.num_agents):
        #         deltaTTCP = self.calc_deltaTTCP(i, j)
        #         Nic = self.calc_Nic(i, j)
        #         other_vel = np.linalg.norm(self.all_agents[j]['v'])
        #         theta = self.calc_theta(i, j)
                
        #         relP = self.calc_relative_positions(i, j, theta)
        #         if relP is None:
        #             px = py = None
        #         # elif theta > 90: # 相手との角度が90度以上の時、Pyは負になる
        #         #     px, py = relP[0], -relP[1]
        #         else:
        #             px, py = relP
                    
        #         self.all_agents[i]['relPx'][self.current_step][j] = px
        #         self.all_agents[i]['relPy'][self.current_step][j] = py     
        #         self.all_agents[i]['theta'][self.current_step][j] = theta
        #         self.all_agents[i]['deltaTTCP'][self.current_step][j] = deltaTTCP
        #         self.all_agents[i]['Nic'][self.current_step][j] = Nic
        #         self.all_agents[i]['all_other_vel'][self.current_step][j] = other_vel
              
        #         if self.prepared_data:
        #             awm = self.awareness_model(i, j, self.awareness_weight)
        #             self.all_agents[i]['awareness'][self.current_step][j] = awm
                
    # 2 
    def set_goals(self, agent: dict[str, np.ndarray]) -> np.ndarray: # [float, float]
        """ 
        ゴールのxy座標の計算
        エージェントが初期速度のまま進んだ場合に通過する、グラフ領域の境界線の座標
        初期位置の値を使うためself.all_agentsではなくself.all_agents2を使う
        ex. self.set_goals(agent=self.all_agents2[10]) -> array([-3.0, -5.0])
        """
        while True:
            
            # x座標がグラフ領域を超える
            if ((agent['p'] + agent['v'])[0] < -FIELD_SIZE):
                # 超えた時の座標をゴールとする
                goal = agent['p'] + agent['v']
                
                # y座標も同時にサイズを超えるかつyが正
                if (goal[1] > FIELD_SIZE - 0.1):
                    # ゴールの座標がグラフ領域の角にならないように調整
                    goal[1] =  goal[1] - 0.1
                    #print("調整入りました")
                    
                # y座標も同時にサイズを超えるかつyが負
                elif (goal[1] < -FIELD_SIZE + 0.1):
                    # ゴールの座標がグラフ領域の角にならないように調整
                    goal[1] = goal[1] + 0.1
                    #print("調整入りました")
                    
                goal[0] = -FIELD_SIZE
                # 端に到達したエージェントを、反対側の端に移動させる
                agent['p'][0] = FIELD_SIZE + ((agent['p'] + agent['v'])[0] + FIELD_SIZE)
                break
                        
            elif ((agent['p'] + agent['v'])[0] > FIELD_SIZE):
                goal = agent['p'] + agent['v']
                
                # y座標も同時にサイズを超えるかつyが正
                if (goal[1] > FIELD_SIZE - 0.1):
                    goal[1] = goal[1] - 0.1
                    #print("調整入りました")

               # y座標も同時にサイズを超えるかつyが負
                elif (goal[1] < -FIELD_SIZE + 0.1):
                    goal[1] = goal[1] + 0.1
                    #print("調整入りました")
                    
                goal[0] = FIELD_SIZE
                agent['p'][0] = -FIELD_SIZE + ((agent['p'] + agent['v'])[0] - FIELD_SIZE)
                break
                
                
            # y座標がグラフ領域を超える
            elif ((agent['p'] + agent['v'])[1] < -FIELD_SIZE):
                # 超えた時の座標をゴールとする
                goal = agent['p'] + agent['v']
                goal[1] = -FIELD_SIZE
                
                agent['p'][1] = FIELD_SIZE + ((agent['p'] + agent['v'])[1] + FIELD_SIZE)
                break
                                        
            elif ((agent['p'] + agent['v'])[1] > FIELD_SIZE):
                goal = agent['p'] + agent['v']
                goal[1] = FIELD_SIZE
                
                agent['p'][1] = -FIELD_SIZE + ((agent['p'] + agent['v'])[1] - FIELD_SIZE)
                break
                
            # エージェントを初期速度のまま動かす
            agent['p'] = agent['p'] + agent['v']

        return goal

        
    # 3 
    def calc_distance_all_agents(self) -> np.array:
        """ 
        全エージェントについて、その他のエージェントとの距離を計算
        """
        dist_all = np.zeros([self.num_agents, self.num_agents])
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                d = self.all_agents[i]['p'] - self.all_agents[j]['p']
                # エージェント間の距離を算出、エージェントのサイズも考慮
                dist_all[i][j] = np.linalg.norm(d) - 2 * self.agent_size
        
        return dist_all

    
    # 4
    def find_visible_agents(self, dist_all: np.ndarray, num: int) -> list[int]:
        """
        エージェント番号numの視野に入った他のエージェントの番号をリストにして返す
        ex. self.find_visible_agents(dist_all, 5) -> [14, 20, 30]
        """
        # near_agentsは360度の視野に入ったエージェント
        # visible_agentsは視野を狭めた場合に視野に入ったエージェント
        near_agents = [i for i, x in enumerate(dist_all[num]) 
                        if x != -(0.2) and x < self.view]
        visible_agents = []
        
        # ゴールベクトルの角度を算出する
        goal_angle = np.degrees(
            fs.calc_rad(self.agent_goals[num][self.goal_count[num]], 
                        self.all_agents[num]['p'])
        )

        for i in near_agents:
            # 近づいたエージェントとの角度を算出
            agent_angle = np.degrees(
                fs.calc_rad(self.all_agents[i]['p'], 
                            self.all_agents[num]['p'])
            )
            
            # 近づいたエージェントとの角度とゴールベクトルの角度の差を計算
            angle_difference = abs(goal_angle - agent_angle)
            
            if angle_difference > 180:
                angle_difference = 360 - angle_difference
                
            # 視野に入っているエージェントをvisible_agentsに追加
            if angle_difference <= self.viewing_angle / 2:
                visible_agents.append(i)
        
        return visible_agents
    
    
    def awareness_model(self, 
                        num: int, 
                        other_num: int, 
                        awareness_weights: AwarenessWeight,
                        debug: bool = False) -> float:
        """
        エージェント番号numの他エージェントに対するawareness modelを計算する(0-1)
        ex. awareness_model(sim, num=10, other_num=15, current_step=20, prepared_data) -> 0.85
        
        説明変数
                deltaTTCP: 自分のTTCPから相手のTTCPを引いた値
                Px: 自分から見た相手の相対位置
                Py: 自分から見た相手の相対位置
                Vself: 自分の歩行速度
                Vother: 相手の歩行速度 
                theta: 自分の向いている方向と相手の位置の角度差 
                Nic: 円内他歩行者数 
                
        被説明変数
                0(注視しない) - 1 (注視する)
        """ 
        agent = self.all_agents[num]
                
        deltaTTCP_mean = self.prepared_data.deltaTTCP_mean
        deltaTTCP_std = self.prepared_data.deltaTTCP_std
        
        Px_mean = self.prepared_data.Px_mean
        Px_std = self.prepared_data.Px_std
        
        Py_mean = self.prepared_data.Py_mean
        Py_std = self.prepared_data.Py_std
        
        Vself_mean = self.prepared_data.Vself_mean
        Vself_std = self.prepared_data.Vself_std
        
        Vother_mean = self.prepared_data.Vother_mean
        Vother_std = self.prepared_data.Vother_std
        
        theta_mean = self.prepared_data.theta_mean
        theta_std = self.prepared_data.theta_std
        
        nic_mean = self.prepared_data.Nic_mean
        nic_std = self.prepared_data.Nic_std
        
        deltaTTCP = (agent['deltaTTCP'][self.current_step][other_num] - deltaTTCP_mean) / deltaTTCP_std
        if np.isnan(deltaTTCP):
            return 0
        Px = (agent['relPx'][self.current_step][other_num] - Px_mean) / Px_std
        Py = (agent['relPy'][self.current_step][other_num] - Py_mean) / Py_std
        Vself = (agent['all_vel'][self.current_step] - Vself_mean) / Vself_std
        Vother = (agent['all_other_vel'][self.current_step][other_num] - Vother_mean) / Vother_std        
        theta = (agent['theta'][self.current_step][other_num] - theta_mean) / theta_std
        Nic = (agent['Nic'][self.current_step][other_num] - nic_mean) / nic_std

        w = awareness_weights
        
        deno = 1 + np.exp(
            -(w.bias + w.deltaTTCP*deltaTTCP + w.Px*Px + w.Py*Py + w.Vself*Vself + \
              w.Vother*Vother + w.theta*theta + w.Nic*Nic)    
        )
        val = 1 / deno
        
        if debug:
            print(f'\ndeltaTTCP {deltaTTCP:.3f}; {deltaTTCP*w.deltaTTCP:.3f}')
            print(f'Px {Px:.3f}; {Px*w.Px:.3f}')
            print(f'Py {Py:.3f}; {Py*w.Py:.3f}')
            print(f'Vself {Vself:.3f}; {Vself*w.Vself:.3f}')
            print(f'Vother {Vother:.3f}; {Vother*w.Vother:.3f}')
            print(f'theta {theta:.3f}; {theta*w.theta:.3f}')
            print(f'Nic {Nic:.3f}; {Nic*w.Nic:.3f}')
            print(f'exp-({w.bias+w.deltaTTCP*deltaTTCP+w.Px*Px+w.Py*Py+w.Vself*Vself+w.Vother*Vother+w.theta*theta+w.Nic*Nic:.3f})')
            print('----------------------------------------------')
            print(f'Awareness {val:.3f}\n')
        else:
            return val
        
    
    def find_agents_to_focus_with_awareness(self, num: int) -> list[int]:
        """
        他エージェントに対してAwarenessモデルを計算し、その値がthresholdより高いエージェント番号をリストにして返す
        """
        agents_to_focus = [] 
        
        for i in range(self.num_agents):
            # -1すると適切な値になったが、検証が必要
            aw = self.all_agents[num]['awareness'][self.current_step][i]
            if aw >= self.awareness:
                agents_to_focus.append(i)
                
        return agents_to_focus
    
    
    # 5
    def simple_avoidance(self, num: int # エージェントの番号
                         ) -> np.ndarray: # [float, float]
        """
        単純な回避ベクトルの生成
        ex. self.simple_avoidance(num=15) -> array([-0.05, 0.02])
        """
        dist_all = self.calc_distance_all_agents()
        visible_agents = self.find_visible_agents(dist_all, num)
        avoid_vec = np.zeros(2)   # 回避ベクトル
        
        if not visible_agents:
            return avoid_vec    
        
        ### the followings are simple vectors ###
        for i in visible_agents:
            # dは視界に入ったエージェントに対して反対方向のベクトル
            d = self.all_agents[num]['p'] - self.all_agents[i]['p']
            d = d / (dist_all[num][i] + 2 * self.agent_size) # 大きさ1のベクトルにする
            d = d * self.simple_avoid_vec # 大きさを固定値にする
            
            if not np.isnan(d).all():
                avoid_vec += d # ベクトルの合成
            
        # # ベクトルの平均を出す
        # ave_avoid_vec = avoid_vec / len(visible_agents)
        
        # return ave_avoid_vec
        
        # ベクトルを平均しない
        return avoid_vec
    
    # 6
    def dynamic_avoidance(self, num: int) -> np.ndarray:
        """
        動的回避ベクトルの生成
        ex. self.dynamic_avoidance(num=15) -> array([-0.05, 0.2])
        """
        dist_all = self.calc_distance_all_agents()
        avoid_vec = np.zeros(2) # 回避ベクトル
        
        # if not self.awareness is False:
        #     visible_agents = self.find_agents_to_focus_with_awareness(num)
        # else:
        #     visible_agents = self.find_visible_agents(dist_all, num)   

        visible_agents = self.find_visible_agents(dist_all, num)

        if not visible_agents:
            return avoid_vec    
                    
        ### the followings are dynamic vectors ###
        for i in visible_agents:
            # 視野の中心にいるエージェントの位置と速度
            self.agents_pos = self.all_agents[num]['p']
            self.agents_vel = self.all_agents[num]['v']
            # 視野に入ったエージェントの位置と速度
            self.visible_agent_pos = self.all_agents[i]['p']
            self.visible_agent_vel = self.all_agents[i]['v']
   
            dist_former = dist_all[num][i]
            
            t = 0
            # 2体のエージェントを1ステップ動かして距離を測定
            self.agents_pos = self.agents_pos + self.agents_vel
            self.visible_agent_pos = self.visible_agent_pos + self.visible_agent_vel
            d = self.agents_pos - self.visible_agent_pos
            dist_latter = np.linalg.norm(d) - 2 * self.agent_size
            
            
            # 視界に入った時が最も近い場合
            if dist_former < dist_latter:
                tcpa = 0
                if dist_former < 0:
                    dcpa = 0 # 最も近い距離で接触している場合はdcpaは0とみなす
                else:
                    dcpa = dist_former * 50 # 単位をピクセルに変換
                    
                    
            # 2者間距離の最小値が出るまでエージェントを動かす
            else:
                while dist_former > dist_latter:
                    dist_former = dist_latter
                    t += self.interval
                    self.agents_pos = self.agents_pos + self.agents_vel
                    self.visible_agent_pos = self.visible_agent_pos + self.visible_agent_vel
                    d = self.agents_pos - self.visible_agent_pos
                    dist_latter = np.linalg.norm(d) - 2 * self.agent_size
                    
                if dist_former < 0:
                    dcpa = 0 # 最も近い距離で接触している場合はdcpaは0とみなす
                else:
                    dcpa = dist_former * 50 # 単位をピクセルに変換
                    
                tcpa = t

            # ブレーキ指標の重み
            w_brake = BrakeWeight()
            a1, b1, c1, d1 = w_brake.a1, w_brake.b1, w_brake.c1, w_brake.d1
            
            # ブレーキ指標の算出
            braking_index = (1 / (1 + np.exp(-c1 - d1 * (tcpa/4000)))) * \
                            (1 / (1 + np.exp(-b1 - a1 * (dcpa/50))))
            
        
            # dは視界に入ったエージェントに対して反対方向のベクトル
            d = self.all_agents[num]['p'] - self.all_agents[i]['p']
            d = d / (dist_all[num][i] + 2 * self.agent_size) # ベクトルの大きさを1にする
            d = d * braking_index # ブレーキ指標の値を反映
            d = d * self.dynamic_avoid_vec # ベクトルの最大値を決定
            
            if not np.isnan(d).all():
                avoid_vec += d # ベクトルの合成
                
                    
        # # ベクトルの平均を出す
        # ave_avoid_vec = avoid_vec / len(visible_agents)

        # return ave_avoid_vec
        
        # ベクトルを平均しない
        return avoid_vec
        


    # 7 
    def calc_Nic(self, num: int, other_num: int) -> int:
        """ 
        エージェント番号numについて、エージェント番号other_numとのNic(Number in the circle)を計算する
        1. エージェントA(num)とエージェントB(other_num)の中点cpを計算し、cpから他の全てのエージェントXとの距離cpXを計算
        2. 1の中で、cpとエージェントAの距離dist_cp_meより小さいcpXの数を計算
        ex. self.calc_Nic(num=10, other_num=15) -> 5
        """ 
        my_pos = self.all_agents[num]['p']
        other_pos = self.all_agents[other_num]['p']
        cp = (my_pos + other_pos) / 2
        radius = np.linalg.norm(cp - my_pos)
        Nic_agents = []
        for i in range(self.num_agents):
            posX = self.all_agents[i]['p']
            dist_cp_posX = np.linalg.norm(posX - cp)
            if (dist_cp_posX <= radius) and (i != num) and (i != other_num):
                Nic_agents.append(i)
        Nic = len(Nic_agents)
    
        return Nic
    
    
    # 8
    def calc_theta(self, num: int, other_num: int) -> float:
        """
        エージェントnumとエージェントother_numのなす角度(radian)を求める
        """
        my_pos_tminus1 = self.all_agents[num]['p'] - self.all_agents[num]['v'] 
        my_pos_t = self.all_agents[num]['p'] 
        extended_end = fs.extend_line(my_pos_tminus1, my_pos_t, length=100)

        other_pos_t = self.all_agents[other_num]['p'] 
                
        theta_deg = fs.calc_angle_two_vectors(my_pos_t, extended_end, other_pos_t)
        
        return theta_deg
            
    
    # 9
    def calc_deltaTTCP(self, num: int, other_num: int) -> float or None:
        my_pos = self.all_agents[num]['p']
        my_vel = self.all_agents[num]['v']
        
        other_pos = self.all_agents[other_num]['p']
        other_vel = self.all_agents[other_num]['v']
        
        deltaTTCP = fs.calc_deltaTTCP(my_pos, my_vel, other_pos, other_vel)
        
        return deltaTTCP
        
    
    def calc_relative_positions(self, 
                                num: int, 
                                other_num: int,
                                theta: float | None) -> np.ndarray or None:
        """
        エージェントnumから見たエージェントother_numの相対位置のxy座標を返す
        """
        if theta is None: # エージェントnum自身に対しての計算の場合、thetaはNoneになる
            return None
        
        arr1 = self.all_agents[num]['p'] - self.all_agents[num]['v']
        arr1_next = self.all_agents[num]['p']

        arr2_next = self.all_agents[other_num]['p']
        
        # 傾きと切片を求める
        # y = ax + b
        # b = y - ax
        # 2直線が垂直 → a*a' = -1 (a' = -(1 / a))
        
        # (1) agent1のPt-1からPtに伸びる直線の傾きと切片
        a1 = (arr1_next[1] - arr1[1]) / (arr1_next[0] - arr1[0])
        b1 = arr1_next[1] - a1 * arr1_next[0]
        
        # (2) (1)に対して垂直な直線の傾きと切片
        a2 = -(1 / a1)
        b2 = arr1_next[1] - a2 * arr1_next[0]
        
        # (3) agent2のPtから(2)に対して垂直な直線の傾きと切片(relPx用)
        a3 = a2
        b3 = arr2_next[1] - a3 * arr2_next[0]
        
        # (4) (1)に並行で、agent2のPtを通る直線の傾きと切片(relPy用)
        a4 = a1
        b4 = arr2_next[1] - a4 * arr2_next[0]
        
        # (5) (1)と(3)の交点
        x1 = -( (b1 - b3) / (a1 - a3) ) 
        y1 = a3 * x1 + b3
        cp1 = np.array([x1, y1])
        relPx = np.linalg.norm(cp1 - arr2_next)
        
        # (6) (2)と(4)の交点
        x2 = -( (b2 - b4) / (a2 - a4) ) 
        y2 = a4 * x2 + b4
        cp2 = np.array([x2, y2])
        relPy = np.linalg.norm(cp2 - arr2_next)
        
        # # 自分の後ろにいるエージェントに対しての距離にはペナルティをつける
        # if theta > 90:
        #     relPy *= 2
            
        relP = np.array([relPx, relPy])
        
        return relP


    # 11
    def record_start_and_goal(self, num: int) -> None:
        """
        完了時間を記録するためのスタート位置とゴール位置を記録
        更新されるパラメータ：
        self.start_pos
        self.goal_pos
        """
        # 前回のゴールが左端にあるとき
        if (self.goal_pos[num][0] == -FIELD_SIZE):
            # スタート位置、ゴール位置を記録
            self.start_pos[num][0] = self.goal_pos[num][0] + 2*FIELD_SIZE
            self.start_pos[num][1] = self.goal_pos[num][1]
            self.goal_pos[num] = self.agent_goals[num][self.goal_count[num]]
            
        # 前回のゴールが右端にあるとき
        elif (self.goal_pos[num][0] == FIELD_SIZE):
            # スタート位置、ゴール位置を記録
            self.start_pos[num][0] = self.goal_pos[num][0] + (-2*FIELD_SIZE)
            self.start_pos[num][1] = self.goal_pos[num][1]
            self.goal_pos[num] = self.agent_goals[num][self.goal_count[num]]
        
        # 前回のゴールが下端にあるとき
        elif (self.goal_pos[num][1] == -FIELD_SIZE):
            # スタート位置、ゴール位置を記録
            self.start_pos[num][0] = self.goal_pos[num][0]
            self.start_pos[num][1] = self.goal_pos[num][1] + 2*FIELD_SIZE
            self.goal_pos[num] = self.agent_goals[num][self.goal_count[num]]
            
        # 前回のゴールが上端にあるとき
        elif (self.goal_pos[num][1] == FIELD_SIZE):
            # スタート位置、ゴール位置を記録
            self.start_pos[num][0] = self.goal_pos[num][0]
            self.start_pos[num][1] = self.goal_pos[num][1] + (-2*FIELD_SIZE)
            self.goal_pos[num] = self.agent_goals[num][self.goal_count[num]]
            
         
    # 12 
    def calc_completion_time(self, num: int) -> float:
        """ 
        ゴールまで到達した際の完了時間を記録
        ex. self.calc_completion_time(num=10) -> 35.6
        """
        # 一回のゴールにおける初めのステップと終わりのステップを記録
        self.start_step[num] = self.goal_step[num] + 1
        self.goal_step[num] = self.current_step
        
        # 一回目のゴール
        if (self.start_step[num] == 1):
            # 一回目のゴールにおける、ゴール位置を記録
            self.goal_pos[num] = self.agent_goals[num][self.goal_count[num]]
            
        # 一回目以降のゴール
        else:
            self.record_start_and_goal(num)
            
        # 完了時間を計算(ゴールまでのステップ/ゴールまでの距離)
        completion_time = (self.goal_step[num] - self.start_step[num] + 1) / \
                          (np.linalg.norm(self.start_pos[num] - self.goal_pos[num]))
        
        # # 外れ値を除外
        # if (completion_time > 200):
        #     #print("消します")
        #     return None
        
        return completion_time
    
    
    # 13 
    def calc_remained_completion_time(self, num: int, step: int) -> float:
        """
        1試行が終わり、ゴールに向かう途中の最後の座標から完了時間を算出(やらなくても良いかもしれない)
        """
        # 初めのステップと終わりのステップを記録
        self.start_step[num] = self.goal_step[num] + 1
        self.goal_step[num] = step
        
        # ゴールする前に境界をはみ出ている場合
        if not (self.goal_temp[num][0] == 0 and self.goal_temp[num][1] == 0):
            # 左右の境界をはみ出た
            if (abs(self.all_agents[num]['p'][0]) > abs(self.all_agents[num]['p'][1])):
                # はみ出る前に戻してあげる
                self.all_agents[num]['p'][0] = -self.all_agents[num]['p'][0]
            # 上下の境界をはみ出た
            elif (abs(self.all_agents[num]['p'][0]) < abs(self.all_agents[num]['p'][1])):
                # はみ出る前に戻してあげる
                self.all_agents[num]['p'][1] = -self.all_agents[num]['p'][1]
            
        self.record_start_and_goal(num)
        
        # スタートからゴールまでの直線の式を計算
        # 傾き
        a = (self.start_pos[num][1] - self.goal_pos[num][1]) / \
            (self.start_pos[num][0] - self.goal_pos[num][0])
        # 切片
        b = -(a * self.start_pos[num][0]) + self.start_pos[num][1]
        
        # エージェントの位置を通り、スタートからゴールまでの直線に垂直な直線の式を計算
        # 傾き
        c = (-1) / a
        # 切片
        d = -(c * self.all_agents[num]['p'][0]) + self.all_agents[num]['p'][1]

        # 2つの直線の交点を算出
        cross_x = (b - d) / (-(a - c))
        cross_y = a * cross_x + b
        cross = np.array([cross_x, cross_y])
        
        # スタートから交点までの距離を計算
        distance = np.linalg.norm(self.start_pos[num] - cross)
        
        # 完了時間を計算(ゴールまでのステップ/ゴールまでの距離)
        completion_time = (self.goal_step[num] - self.start_step[num] + 1) / distance

        # if (completion_time > 200 or completion_time < 10):
        #     #print("消しました")
        #     return None
        
        return completion_time


    # 14
    def check_if_goaled(self) -> None:
        """
        各エージェントがゴールに到達したかどうかをチェックする
        ゴールしていた場合、完了時間の算出、ゴールカウントの更新、はみ出た時用のゴールの初期化を行う
        更新されるパラメータ：
        self.completion_time
        self.goal_count
        self.goal_tmp
        self.all_agents
        """
        for i in range(self.num_agents):
            # x座標が左端をこえる
            if ((self.all_agents[i]['p'] + self.all_agents[i]['v'])[0] < -FIELD_SIZE):
                # ゴールに到着
                if (self.all_agents[i]['p'][0] > 
                   self.agent_goals[i][self.goal_count[i]][0] - 0.1 
                   and 
                   self.all_agents[i]['p'][0] < 
                   self.agent_goals[i][self.goal_count[i]][0] + 0.1
                   and 
                   self.all_agents[i]['p'][1] > 
                   self.agent_goals[i][self.goal_count[i]][1] - 0.1 
                   and 
                   self.all_agents[i]['p'][1] < 
                   self.agent_goals[i][self.goal_count[i]][1] + 0.1):
                    
                    # 通常のゴールに到着
                    if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                        # 完了時間を算出
                        completion_time = self.calc_completion_time(i)
                        if not completion_time == None:
                            self.completion_time.append(completion_time)
                        # ゴールした回数を更新
                        self.goal_count[i] = self.goal_count[i] + 1
                        
                    # はみ出た時用のゴールに到着
                    else:
                        # はみ出た時用のゴールを初期化
                        self.goal_temp[i][0] = 0
                        self.goal_temp[i][1] = 0
                        
                # ゴールに到着せず、境界を超える  
                else:
                    # はみ出た時用のゴールが設定されていない
                    if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                        # はみ出た時用のゴールを設定
                        self.goal_temp[i][0] = self.agent_goals[i][self.goal_count[i]][0] + 2*FIELD_SIZE
                        self.goal_temp[i][1] = self.agent_goals[i][self.goal_count[i]][1]
                        
                    # はみ出た時用のゴールが設定されている
                    else:
                        # はみ出た時用のゴールを初期化
                        self.goal_temp[i][0] = 0
                        self.goal_temp[i][1] = 0
                
                # エージェントを反対の端へ移動
                self.all_agents[i]['p'][0] = FIELD_SIZE + (
                    (self.all_agents[i]['p'] + self.all_agents[i]['v'])[0] + FIELD_SIZE
                )
        
            
            # x座標が右端をこえる
            elif ((self.all_agents[i]['p']+self.all_agents[i]['v'])[0] > FIELD_SIZE):
                
                # ゴール判定
                if (self.all_agents[i]['p'][0] > 
                   self.agent_goals[i][self.goal_count[i]][0] - 0.1 
                   and 
                   self.all_agents[i]['p'][0] < 
                   self.agent_goals[i][self.goal_count[i]][0] + 0.1 
                   and 
                   self.all_agents[i]['p'][1] > 
                   self.agent_goals[i][self.goal_count[i]][1] - 0.1 
                   and 
                   self.all_agents[i]['p'][1] < 
                   self.agent_goals[i][self.goal_count[i]][1] + 0.1):
                    
                    # ゴールが調整されているか確認
                    if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                        # 完了時間を算出
                        completion_time = self.calc_completion_time(i)
                        if not completion_time == None:
                            self.completion_time.append(completion_time)
                        self.goal_count[i] = self.goal_count[i] + 1
                    else:
                        # はみ出た時用のゴールを初期化
                        self.goal_temp[i][0] = 0
                        self.goal_temp[i][1] = 0
                        
                else:
                    # ゴールが調整されているか確認
                    if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                        # 境界をこえた用のゴールを設定
                        self.goal_temp[i][0] = self.agent_goals[i][self.goal_count[i]][0] + (-2*FIELD_SIZE)
                        self.goal_temp[i][1] = self.agent_goals[i][self.goal_count[i]][1]
                    else:
                        # はみ出た時用のゴールを初期化
                        self.goal_temp[i][0] = 0
                        self.goal_temp[i][1] = 0
                        
                self.all_agents[i]['p'][0] = -FIELD_SIZE + (
                    (self.all_agents[i]['p'] + self.all_agents[i]['v'])[0] - FIELD_SIZE
                )

                
            # y座標が下をこえる
            elif ((self.all_agents[i]['p']+self.all_agents[i]['v'])[1] < -FIELD_SIZE):
                
                # ゴール判定
                if (self.all_agents[i]['p'][0] > 
                   self.agent_goals[i][self.goal_count[i]][0] - 0.1 
                   and 
                   self.all_agents[i]['p'][0] < 
                   self.agent_goals[i][self.goal_count[i]][0] + 0.1 
                   and 
                   self.all_agents[i]['p'][1] > 
                   self.agent_goals[i][self.goal_count[i]][1] - 0.1 
                   and 
                   self.all_agents[i]['p'][1] < 
                   self.agent_goals[i][self.goal_count[i]][1] + 0.1):
                    
                    # ゴールが調整されているか確認
                    if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                        # 完了時間を算出
                        completion_time = self.calc_completion_time(i)
                        if not completion_time == None:
                            self.completion_time.append(completion_time)
                        self.goal_count[i] = self.goal_count[i] + 1
                    else:
                        # はみ出た時用のゴールを初期化
                        self.goal_temp[i][0] = 0
                        self.goal_temp[i][1] = 0
                        
                else:
                    # ゴールが調整されているか確認
                    if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                        # 境界をこえた用のゴールを設定
                        self.goal_temp[i][0] = self.agent_goals[i][self.goal_count[i]][0]
                        self.goal_temp[i][1] = self.agent_goals[i][self.goal_count[i]][1] + 2*FIELD_SIZE
                    else:        
                        # はみ出た時用のゴールを初期化
                        self.goal_temp[i][0] = 0
                        self.goal_temp[i][1] = 0
                        
                self.all_agents[i]['p'][1] = FIELD_SIZE + (
                    (self.all_agents[i]['p']+self.all_agents[i]['v'])[1] + FIELD_SIZE
                )
                
            # y座標が上をこえる     
            elif ((self.all_agents[i]['p']+self.all_agents[i]['v'])[1] > FIELD_SIZE):
                
                # ゴール判定
                if (self.all_agents[i]['p'][0] > 
                   self.agent_goals[i][self.goal_count[i]][0] - 0.1 
                   and 
                   self.all_agents[i]['p'][0] < 
                   self.agent_goals[i][self.goal_count[i]][0] + 0.1 
                   and 
                   self.all_agents[i]['p'][1] > 
                   self.agent_goals[i][self.goal_count[i]][1] - 0.1 
                   and 
                   self.all_agents[i]['p'][1] < 
                   self.agent_goals[i][self.goal_count[i]][1] + 0.1):
                    
                    # ゴールが調整されているか確認
                    if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                        # 完了時間を算出
                        completion_time = self.calc_completion_time(i)
                        if not completion_time == None:
                            self.completion_time.append(completion_time)
                        self.goal_count[i] = self.goal_count[i] + 1
                    else:
                        # はみ出た時用のゴールを初期化
                        self.goal_temp[i][0] = 0
                        self.goal_temp[i][1] = 0
                        
                else:
                    # ゴールが調整されているか確認
                    if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                        # 境界をこえた用のゴールを設定
                        self.goal_temp[i][0] = self.agent_goals[i][self.goal_count[i]][0]
                        self.goal_temp[i][1] = self.agent_goals[i][self.goal_count[i]][1] + (-2*FIELD_SIZE)
                    else:
                        # はみ出た時用のゴールを初期化
                        self.goal_temp[i][0] = 0
                        self.goal_temp[i][1] = 0
                        

                self.all_agents[i]['p'][1] = -FIELD_SIZE + (
                    (self.all_agents[i]['p']+self.all_agents[i]['v'])[1] - FIELD_SIZE
                )

    
    # 15 
    def move_agents(self) -> None:
        """
        エージェントを動かす
        更新されるパラメータ：
        self.all_agents
        """
        #self.current_step += 1
        for i in range(self.num_agents):
            # はみ出た時用のゴールが設定されていない
            # 通常のゴールに向かうベクトルと、回避ベクトルを足したものが速度になる
            if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                self.all_agents[i]['v'] = fs.rotate_vec(
                    np.array([self.goal_vec, 0]), 
                    fs.calc_rad(self.agent_goals[i][self.goal_count[i]],
                                self.all_agents[i]['p'])
                )     

            # はみ出た時用のゴールが設定されている
            # はみ出た時用のゴールに向かうベクトルと、回避ベクトルを足したものが速度になる
            else:
                self.all_agents[i]['v'] = fs.rotate_vec(
                    np.array([self.goal_vec, 0]), 
                    fs.calc_rad(self.goal_temp[i], self.all_agents[i]['p'])
                ) 
                
                
            # 回避ベクトルを足す
            if self.all_agents[i]['avoidance'] == 0: # 単純回避ベクトルを足す
                self.all_agents[i]['v'] += self.simple_avoidance(i)
                
            elif self.all_agents[i]['avoidance'] == 1: # 動的回避ベクトルを足す
                self.all_agents[i]['v'] += self.dynamic_avoidance(i)
                
        self.current_step += 1
        self.check_if_goaled()
        
        for i in range(self.num_agents):
            # 移動後の座標を確定      
            self.all_agents[i]['p'] += self.all_agents[i]['v']
                    
        self.update_parameters()     
        
        row = self.record_agent_information()
        self.data.append(row)
        
    # 16
    def simulate(self) -> None:
        """
        self.num_stepsの回数だけエージェントを動かす
        シミュレーションを行うときに最終的に呼び出すメソッド
        """
        self.disp_info()
        start = time.perf_counter() # 実行時間を結果csvに記録する
        
        for t in tqdm(range(self.num_steps)):
            self.move_agents()
            
        end = time.perf_counter()
        self.exe_time = end - start
        
    # 18
    def plot_positions_aware(self, num, step: int, awareness_weight) -> None:
        """
        各エージェントの座標を、エージェントの番号付きでプロットし、エージェント番号numのAwareness modelを計算する
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        #s = 45
        for i in range(self.num_agents):
            color = 'green' if i == num else 'blue'
            
            pos_px = (self.all_agents[i]['all_pos'][step]*50)+250
            ax.add_artist(Circle(pos_px, radius=5, color=color, alpha=0.6))
            # ax.scatter(*self.all_agents[i]['all_pos'][step],
            #             color=color, alpha=0.6, s=s)
            ax.annotate(i, xy=pos_px+1.5)
            if not step == 0:
                tminus1_pos = (self.all_agents[i]['all_pos'][step-1]*50)+250
                ax.add_artist(Circle((tminus1_pos), radius=5, color=color, alpha=0.2))
                #ax.scatter(*tminus1_pos, color=color, alpha=0.2, s=s)
        
        awms = []
        for i in range(self.num_agents):
            awm = self.awareness_model(num, i, awareness_weight, debug=False)
            if awm is not None and awm >= 0.8:
                awms.append([i, awm])
        
        my_posx = (self.all_agents[num]['all_pos'][step][0]*50)+250
        my_posy = (self.all_agents[num]['all_pos'][step][1]*50)+250
        
        for i in awms:
            other_posx = (self.all_agents[i[0]]['all_pos'][step][0]*50)+250
            other_posy = (self.all_agents[i[0]]['all_pos'][step][1]*50)+250
            
            ax.arrow(x=my_posx, y=my_posy, dx=other_posx-my_posx, dy=other_posy-my_posy,
                      color='tab:blue', alpha=0.5)
            
        ax.set_xlim(0, 500)
        ax.set_ylim(0, 500)
        ax.set_xticks(range(0, 501, 50))
        ax.set_yticks(range(0, 501, 50))
        ax.grid()
        plt.show()    
        
        
    # 18 
    def approach_detect(self, dist: float) -> np.ndarray: 
        """ 
        指定した距離より接近したエージェントの数を返す
        ex. self.approach_detect(dist=0.5) -> array([[0, 3],[1, 2],...[24, 1]])
        """
        dist_all = self.calc_distance_all_agents()
        approach_agent = []
        
        # それぞれのエージェントについて、distより接近したエージェントの数を記録
        for t in range(self.num_agents):
            visible_agents = [i for i, x in enumerate(dist_all[t]) 
                              if x != -(0.2) and x < dist]
            approach_agent.append([t, len(visible_agents)])
        approach_agent = np.array(approach_agent) 
            
        return approach_agent
    
        
    # 19
    def record_agent_information(self) -> np.ndarray:
        """
        全エージェントの位置と速度、接近を記録
        """
        # 初期の位置と速度を記録
        row = np.concatenate([self.all_agents[0]['p'], self.all_agents[0]['v']])
        # rowにはある時刻の全エージェントの位置と速度が入る
        for i in range(1, self.num_agents):
            row = np.concatenate([row, self.all_agents[i]['p'], self.all_agents[i]['v']])
            
        # エージェントの接近を記録
        # 衝突したエージェント
        collision_agent = self.approach_detect(0)
        for i in range(self.num_agents):
            row = np.append(row, collision_agent[i][1])

        # 視野の半分まで接近したエージェント
        collision_agent = self.approach_detect(0.5)
        for i in range(self.num_agents):
            row = np.append(row, collision_agent[i][1])
            
        # 視野の4分の1まで接近したエージェント
        collision_agent = self.approach_detect(0.25)
        for i in range(self.num_agents):
            row = np.append(row, collision_agent[i][1])

        # 視野の8分の1まで接近したエージェント
        collision_agent = self.approach_detect(0.125)
        for i in range(self.num_agents):
            row = np.append(row, collision_agent[i][1])
            
        return row
    
        
    # 20
    def record_approaches(self, 
                          approach_dist: Literal['collision','half','quarter','one_eigth']
                          ) -> list[int]:
        """
        エージェント同士が接近した回数を、接近度合い別で出力する
        ex. self.record_approaches('collision', STEP=500, data) -> [0,0,3,...,12]
        """
        if approach_dist == 'collision':
            start, stop = 4*self.num_agents, 5*self.num_agents
            
        elif approach_dist == 'half':
            start, stop = 5*self.num_agents, 6*self.num_agents
            
        elif approach_dist == 'quarter':
            start, stop = 6*self.num_agents, 7*self.num_agents
            
        elif approach_dist == 'one_eigth':
            start, stop = 7*self.num_agents, 8*self.num_agents
            
        approach = []
        for i in range(start, stop, 1):
            total = 0
            for j in range(self.num_steps):
                # 一試行で何回エージェントに衝突したか
                total += self.data[j+1][i]
            
            # 全エージェントの衝突した回数を記録
            approach.append(total)
        
        return approach
        
    
    # 21
    def return_results_as_df(self) -> pd.core.frame.DataFrame:
        """
        1試行の記録をデータフレームにして返す
        """
        # この関数は破壊的操作で1回目と2回目の値が変化するため、2回目以降の呼び出しを禁止する
        assert self.returned_results == False, "Results have been already returend."
        
        # 最後の座標から完了時間を算出
        for i in range(self.num_agents):
            last_completion_time = self.calc_remained_completion_time(i, self.num_steps)
            if not last_completion_time == None:
                self.completion_time.append(last_completion_time)
       
        # 衝突した数、視野の半分、視野の四分の一、視野の八分の一に接近した回数
        collision = self.record_approaches('collision')
        half =  self.record_approaches('half')
        quarter =  self.record_approaches('quarter')
        one_eighth =  self.record_approaches('one_eigth')
        
        # 各指標の平均を計算
        collision_mean = np.mean(collision)
        half_mean = np.mean(half)
        quarter_mean = np.mean(quarter)
        one_eighth_mean = np.mean(one_eighth)
        completion_time_mean = np.mean(self.completion_time)

        # 結果のデータを保存
        dict_result = {'time': completion_time_mean,
                       'half': half_mean,
                       'quarter': quarter_mean,
                       'one_eigth': one_eighth_mean,
                       'collision': collision_mean,
                       'agent': self.num_agents,
                       'viewing_angle': self.viewing_angle,
                       'num_steps': self.num_steps,
                       'dynamic_percent': self.dynamic_percent,
                       'simple_avoid_vec': self.simple_avoid_vec,
                       'awareness': self.awareness,
                       'sum_goal_count': np.sum(self.goal_count),
                       'exe_time_second': np.round(self.exe_time, 3),
                       'exe_time_min': np.round(self.exe_time/60, 3)}
        
        df_result = pd.DataFrame(dict_result, index=[f'seed_{self.random_seed}'])
        self.returned_results = True
        
        return df_result
    

    def save_data_for_awareness(self, 
                                save_as: str = 'data_for_awareness.npz',
                                return_dict: bool = False) -> None or dict:
        """
        Awarenessモデルの計算時の平均、標準偏差のためのデータを保存する
        return_dictがTrueのとき、データの保存は行わない
        """
        all_data = {
            'all_deltaTTCP': np.array(
                [self.all_agents[i]['deltaTTCP'] for i in range(self.num_agents)]
            ),
            'all_Px': np.array(
                [self.all_agents[i]['relPx'] for i in range(self.num_agents)]
            ),
            'all_Py': np.array(
                [self.all_agents[i]['relPy'] for i in range(self.num_agents)]
            ),
            'all_Vself': np.array(
                [self.all_agents[i]['all_vel'] for i in range(self.num_agents)]
            ),
            'all_Vother': np.array(
                [self.all_agents[i]['all_other_vel'] for i in range(self.num_agents)]
            ),
            'all_theta': np.array(
                [self.all_agents[i]['theta'] for i in range(self.num_agents)]
            ),
            'all_Nic': np.array(
                [self.all_agents[i]['Nic'] for i in range(self.num_agents)]
            )    
        }        
        if return_dict:
            return all_data
        else:
            np.savez_compressed(save_as, **all_data)
        
