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
9. calc_completion_time(num, current_step)
10. calc_last_completion_time(num)
11. check_if_goaled(current_step)
12. simulate(current_step)
13. show_image()
14. plot_positions()
15. approach_detect(dist)
16. record_agent_information()
17. record_approaches(approach_dist, step, sim_data)
"""

# %% import libraries
from copy import deepcopy
from typing import Literal
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import funcSimulation as fs

# %% 
# グラフの目盛りの最大値・最小値
FIELD_SIZE = 5 
# 目盛りは最大値5、最小値-5で10目盛り
# グラフ領域の幅と高さは500pxなので、1pxあたり0.02目盛りとなる

# 妨害指標の4係数は標準化したやつを使う
abcd = {'a1': -5.145, # -0.034298
        'b1': 3.348, # 3.348394
        'c1': 4.286, # 4.252840
        'd1': -13.689} # -0.003423

# a1: -5.145 (-0.034298)
# b1: 3.348 (3.348394)
# c1: 4.286 (4.252840)
# d1: -13.689 (-0.003423)

class Agent:
    def __init__(self, avoidance, viewing_angle):
        self.avoidance = avoidance
        self.viewing_angle = viewing_angle

# %% シミュレーションに関わるクラス
class Simulation:
    def __init__(self, 
                 warmup: int = 50,
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
                 random_seed: int = 0):
        
        self.warmup = warmup # 値の標準化のために値を保存するwarmupのステップ数
        self.num_steps = num_steps
        self.interval = interval # 100msごとにグラフを更新してアニメーションを作成
        self.num_agents_size = agent_size # エージェントの半径(目盛り) = 5px
        self.num_agents = num_agents # エージェント数
        self.view = view # 視野の半径(目盛り) = 50px:エージェント5体分
        self.viewing_angle = viewing_angle # 視野の角度
        self.goal_vec = goal_vec # ゴールベクトルの大きさ(目盛り)
        self.dynamic_percent = dynamic_percent # 動的回避を基に回避するエージェントの割合
        self.simple_avoid_vec = simple_avoid_vec # 単純回避での回避ベクトルの大きさ(目盛り)
        self.dynamic_avoid_vec = dynamic_avoid_vec # 動的回避での回避ベクトルの最大値(目盛り)
        self.random_seed = random_seed

        self.all_agents = [] # 全エージェントの座標を記録
        self.all_agents2 = [] # ゴールの計算用
        self.first_agent = [] # 初期位置記録用
        self.num_agents_goal = []
        self.first_pos =[]
        
        # 動的回避を行うエージェントの数
        self.num_dynamic_agent = int(np.round(self.num_agents*self.dynamic_percent))
        
        # エージェントの生成
        np.random.seed(self.random_seed)
        for n in range(self.num_agents):
            # グラフ領域の中からランダムに座標を決定
            pos = np.random.uniform(-FIELD_SIZE, FIELD_SIZE, 2)
            vel = np.random.uniform(-FIELD_SIZE, FIELD_SIZE, 2)
            
            if n < self.num_dynamic_agent:
                avoidance = 'dynamic'
            else:
                avoidance = 'simple'
            
            # 座標(0, 0)から座標velへのベクトルがエージェントの初期速度になる
            # self.all_agentsの1つの要素に1体のエージェントの位置と速度が記録
            self.all_agents.append(
                {'avoidance': avoidance, 
                 'p': pos, 
                 'v': fs.rotate_vec(np.array([self.goal_vec, 0]), 
                                    fs.calc_rad(vel, np.array([0, 0])))
                 }
            )
            
        # 毎時点での位置xyを全て記録する
        # self.all_pos[0]の0は初期位置を表す
        # 例えば5ステップ目の記録はself.all_pos[5]
        self.all_pos = np.zeros([num_steps+1, 2, self.num_agents])
        self.all_vel = np.zeros([num_steps+1, self.num_agents])
        self.record_parameters(0)                                             
            
        # 初期位置と初期速度をコピー
        self.all_agents2 = deepcopy(self.all_agents)
        self.first_agent = deepcopy(self.all_agents)
        
        # エージェントの初期位置を保存
        for i in range(self.num_agents):
            self.first_pos.append(self.first_agent[i]['p'])
        
        # エージェントにゴールを8ずつ設定
        for i in range(self.num_agents):
            goals = [self.set_goals(self.all_agents2[i]) for _ in range(8)]
            self.num_agents_goal.append(goals)
            
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
        
        # 標準化のためにwarmup期間の値を保存
        # warmup終了後はウィンドウを1ステップずつずらしながら値を更新する
        self.remain_warmup = warmup
        self.all_theta = []
        self.all_px = []
        self.all_py = []
        self.all_Nic = [] # とりあえずNicは無視する
        self.all_deltaTTCP = []
        
        
    # def update_vals_for_standardize(self):
    #     for i, _ in enumerate(self.all_agents):
    #         vels = np.linalg.norm(self.all_agents[i]['v'])
    #         self.all_velocity.append(vels)
    #         px = self.all_agents[i]['p'][0]
    #         self.all_px.append(px)
    #         py = self.all_agents[i]['p'][1]
    #         self.all_py.append(py)
    #         # calculate theta
    #         for j, _ in enumerate(self.all_agents):
    #             other_pos = self.all_agents[j]['p']
            
    #     if self.remain_warmup > 0:
    #         self.remain_warmup -= 1
    #     else:
    #         self.all_velocity = self.all_velocity[len(self.all_agents):]
    
    def record_parameters(self, current_step: int) -> None:
        for i in range(self.num_agents):
            self.all_pos[current_step][0][i] = self.all_agents[i]['p'][0] # x
            self.all_pos[current_step][1][i] = self.all_agents[i]['p'][1] # y
            
            vel = np.linalg.norm(self.all_agents[i]['v'])
            self.all_vel[current_step][i] = vel
    
    # 1. 
    def set_goals(self, agent: dict[str, np.array]) -> np.array: # [float, float]
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

        
    # 2. 
    def calc_distance_all_agents(self) -> np.array:
        """ 
        全エージェントについて、その他のエージェントとの距離を計算
        """
        dist_all = np.zeros([self.num_agents, self.num_agents])
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                d = self.all_agents[i]['p'] - self.all_agents[j]['p']
                # エージェント間の距離を算出、エージェントのサイズも考慮
                dist_all[i][j] = np.linalg.norm(d) - 2 * self.num_agents_size
        
        return dist_all
    
    
    # 3.
    def find_visible_agents(self, dist_all: np.array, num: int) -> list[int]:
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
            fs.calc_rad(self.num_agents_goal[num][self.goal_count[num]], 
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
    
    
    # 4. 
    def simple_avoidance_with_focus(self, num: int # エージェントの番号
                                    ) -> np.array: # [float, float]
        """
        単純な回避ベクトルの生成(awareness modelで計算相手を選定)
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
            d = d / (dist_all[num][i] + 2 * self.num_agents_size) # 大きさ1のベクトルにする
            d = d * self.simple_avoid_vec # 大きさを固定値にする
            
            avoid_vec += d # 回避ベクトルを合成する
            
        # ベクトルの平均を出す
        return avoid_vec / len(visible_agents)
    
    
    # 5. 
    def dynamic_avoidance_with_focus(self, num: int) -> np.array:
        """
        動的回避ベクトルの生成(awareness modelで計算相手を選定)
        ex. self.dynamic_avoidance(num=15) -> array([-0.05, 0.2])
        """
        dist_all = self.calc_distance_all_agents()
        visible_agents = self.find_visible_agents(dist_all, num)
        avoid_vec = np.zeros(2)   # 回避ベクトル
        
        if not visible_agents:
            return avoid_vec  
            
        ### the followings are dynamic vectors ###
        for i in visible_agents:
            # 視野の中心にいるエージェントの位置と速度
            self.num_agents_pos = self.all_agents[num]['p']
            self.num_agents_vel = self.all_agents[num]['v']
            # 視野に入ったエージェントの位置と速度
            self.visible_agent_pos = self.all_agents[i]['p']
            self.visible_agent_vel = self.all_agents[i]['v']

            
            dist_former = dist_all[num][i]
            
            t = 0
            # 2体のエージェントを1ステップ動かして距離を測定
            self.num_agents_pos = self.num_agents_pos + self.num_agents_vel
            self.visible_agent_pos = self.visible_agent_pos + self.visible_agent_vel
            d = self.num_agents_pos - self.visible_agent_pos
            dist_latter = np.linalg.norm(d) - 2 * self.num_agents_size
            
            
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
                    self.num_agents_pos = self.num_agents_pos + self.num_agents_vel
                    self.visible_agent_pos = self.visible_agent_pos + self.visible_agent_vel
                    d = self.num_agents_pos - self.visible_agent_pos
                    dist_latter = np.linalg.norm(d) - 2 * self.num_agents_size
                    
                if dist_former < 0:
                    dcpa = 0 # 最も近い距離で接触している場合はdcpaは0とみなす
                else:
                    dcpa = dist_former * 50 # 単位をピクセルに変換
                    
                tcpa = t

            a1, b1, c1, d1 = abcd['a1'], abcd['b1'], abcd['c1'], abcd['d1']
            # ブレーキ指標の算出
            braking_index = (1 / (1 + np.exp(-c1 - d1 * (tcpa/4000)))) * \
                            (1 / (1 + np.exp(-b1 - a1 * (dcpa/50))))
            
            # dは視界に入ったエージェントに対して反対方向のベクトル
            d = self.all_agents[num]['p'] - self.all_agents[i]['p']
            d = d / (dist_all[num][i] + 2 * self.num_agents_size) # ベクトルの大きさを1にする
            d = d * braking_index # ブレーキ指標の値を反映
            d = d * self.dynamic_avoid_vec # ベクトルの最大値を決定
            
            avoid_vec += d # ベクトルの合成
    
        # ベクトルの平均を出す
        return avoid_vec / len(visible_agents)
    
    
    # 6. 
    def calc_Nic(self, num: int) -> np.array:
        """ 
        エージェント番号numについて、Nic(Number in the circle)を計算する
        1. エージェントA(num)とエージェントBの中点cpを計算し、cpから他の全てのエージェントXとの距離cpxを計算
        2. 1の中で、cpとエージェントAの距離dist_cp_meより小さいcpxの数を計算
        ex. self.calc_Nic(num=10) -> array([[0,1], [1,2],...,[24,12]]) (相手,Nic)
        """ 
        all_Nics = []
        posA = self.all_agents[num]['p']
        # iは半径を計算するエージェントで、jはその他のエージェント
        for i in range(self.num_agents):
            posB = self.all_agents[i]['p']
            cp = (posA + posB) / 2
            radius = np.linalg.norm(cp - posA)
            
            Nic_agents = []
            for j in range(self.num_agents):
                posX = self.all_agents[j]['p']
                dist_cp_posX = np.linalg.norm(posX - cp)
                if (dist_cp_posX <= radius) and (j != num) and (j != i):
                    Nic_agents.append(i)
            all_Nics.append([i, len(Nic_agents)]) 
        all_Nics = np.array(all_Nics)
        
        return all_Nics
    
    
    # 7. 
    def find_agents_to_focus(self, num: int) -> np.array:
        """
        awareness modelを用いて、注視相手を選定する
        ex. self.find_agents_to_focus(
            num=10) -> array([[0, 0.2], [1, 0.3],...,[24, 0.8]]) (相手,awareness)
        """ 
        agents_to_focus = []
        my_posx, my_posy = self.all_agents[num]['p']
        vselfxy = self.all_agents[num]['v']
        Vself = np.linalg.norm(vselfxy)
        
        # 時点t-1から時点tのxy座標への直線line1
        a = self.all_agents[num]['p'] + self.all_agents[num]['v']
        b = self.all_agents[num]['p']
        line1 = np.array([b, a])
        
        for i in range(self.num_agents):
            other_posx, other_posy = self.all_agents[i]['p']
            Px = other_posx - my_posx
            Py = other_posy - my_posy
            votherxy = self.all_agents[i]['v']
            Vother = np.linalg.norm(votherxy)
            
            c = self.all_agents[i]['p']
            line2 = np.array([b, c])
            theta = fs.calc_angle_two_lines(line1, line2)
            
            # deltaTTCPが計算できるときはawmを返し、そうでなければ0を返す
            try:
                deltaTTCP = fs.calc_deltaTTCP(*vselfxy, my_posx, my_posy, 
                                              *votherxy, other_posx, other_posy)
                Nic = self.calc_Nic(num)[i][1]
                awm = fs.awareness_model(deltaTTCP, Px, Py, Vself, Vother, theta, Nic)
                agents_to_focus.append([i, awm])
            except TypeError:
                agents_to_focus.append([i, 0])
        agents_to_focus = np.array(agents_to_focus)
        
        return agents_to_focus
    
    
    # 8.
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
            self.goal_pos[num] = self.num_agents_goal[num][self.goal_count[num]]
            
        # 前回のゴールが右端にあるとき
        elif (self.goal_pos[num][0] == FIELD_SIZE):
            # スタート位置、ゴール位置を記録
            self.start_pos[num][0] = self.goal_pos[num][0] + (-2*FIELD_SIZE)
            self.start_pos[num][1] = self.goal_pos[num][1]
            self.goal_pos[num] = self.num_agents_goal[num][self.goal_count[num]]
        
        # 前回のゴールが下端にあるとき
        elif (self.goal_pos[num][1] == -FIELD_SIZE):
            # スタート位置、ゴール位置を記録
            self.start_pos[num][0] = self.goal_pos[num][0]
            self.start_pos[num][1] = self.goal_pos[num][1] + 2*FIELD_SIZE
            self.goal_pos[num] = self.num_agents_goal[num][self.goal_count[num]]
            
        # 前回のゴールが上端にあるとき
        elif (self.goal_pos[num][1] == FIELD_SIZE):
            # スタート位置、ゴール位置を記録
            self.start_pos[num][0] = self.goal_pos[num][0]
            self.start_pos[num][1] = self.goal_pos[num][1] + (-2*FIELD_SIZE)
            self.goal_pos[num] = self.num_agents_goal[num][self.goal_count[num]]
            
         
    # 9. 
    def calc_completion_time(self, num: int, current_step: int) -> float:
        """ 
        ゴールまで到達した際の完了時間を記録
        ex. self.calc_completion_time(num=10, current_step=10) -> 35.6
        """
        # 一回のゴールにおける初めのステップと終わりのステップを記録
        self.start_step[num] = self.goal_step[num] + 1
        self.goal_step[num] = current_step
        
        # 一回目のゴール
        if (self.start_step[num] == 1):
            # 一回目のゴールにおける、ゴール位置を記録
            self.goal_pos[num] = self.num_agents_goal[num][self.goal_count[num]]
            
        # 一回目以降のゴール
        else:
            self.record_start_and_goal(num)
            
        # 完了時間を計算(ゴールまでのステップ/ゴールまでの距離)
        completion_time = (self.goal_step[num] - self.start_step[num] + 1) / \
                          (np.linalg.norm(self.start_pos[num] - self.goal_pos[num]))
        
        # 外れ値を除外
        if (completion_time > 200):
            #print("消します")
            print(completion_time)
            return None
        
        return completion_time
    
    
    # 10. 
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

        if (completion_time > 200 or completion_time < 10):
            #print("消しました")
            print(completion_time)
            return None
        
        return completion_time


    # 11.
    def check_if_goaled(self, current_step: int) -> None:
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
                   self.num_agents_goal[i][self.goal_count[i]][0] - 0.1 
                   and 
                   self.all_agents[i]['p'][0] < 
                   self.num_agents_goal[i][self.goal_count[i]][0] + 0.1
                   and 
                   self.all_agents[i]['p'][1] > 
                   self.num_agents_goal[i][self.goal_count[i]][1] - 0.1 
                   and 
                   self.all_agents[i]['p'][1] < 
                   self.num_agents_goal[i][self.goal_count[i]][1] + 0.1):
                    
                    # 通常のゴールに到着
                    if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                        # 完了時間を算出
                        completion_time = self.calc_completion_time(i, current_step)
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
                        self.goal_temp[i][0] = self.num_agents_goal[i][self.goal_count[i]][0] + 2*FIELD_SIZE
                        self.goal_temp[i][1] = self.num_agents_goal[i][self.goal_count[i]][1]
                        
                    # はみ出た時用のゴールが設定されている
                    else:
                        # はみ出た時用のゴールを初期化
                        self.goal_temp[i][0] = 0
                        self.goal_temp[i][1] = 0
                
                # エージェントを反対の端へ移動
                self.all_agents[i]['p'][0] = FIELD_SIZE + (
                    (self.all_agents[i]['p']+self.all_agents[i]['v'])[0] + FIELD_SIZE
                )
        
            
            # x座標が右端をこえる
            elif ((self.all_agents[i]['p']+self.all_agents[i]['v'])[0] > FIELD_SIZE):
                
                # ゴール判定
                if (self.all_agents[i]['p'][0] > 
                   self.num_agents_goal[i][self.goal_count[i]][0] - 0.1 
                   and 
                   self.all_agents[i]['p'][0] < 
                   self.num_agents_goal[i][self.goal_count[i]][0] + 0.1 
                   and 
                   self.all_agents[i]['p'][1] > 
                   self.num_agents_goal[i][self.goal_count[i]][1] - 0.1 
                   and 
                   self.all_agents[i]['p'][1] < 
                   self.num_agents_goal[i][self.goal_count[i]][1] + 0.1):
                    
                    # ゴールが調整されているか確認
                    if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                        # 完了時間を算出
                        completion_time = self.calc_completion_time(i, current_step)
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
                        self.goal_temp[i][0] = self.num_agents_goal[i][self.goal_count[i]][0] + (-2*FIELD_SIZE)
                        self.goal_temp[i][1] = self.num_agents_goal[i][self.goal_count[i]][1]
                    else:
                        # はみ出た時用のゴールを初期化
                        self.goal_temp[i][0] = 0
                        self.goal_temp[i][1] = 0
                        
                self.all_agents[i]['p'][0] = -FIELD_SIZE + \
                    ((self.all_agents[i]['p']+self.all_agents[i]['v'])[0] - FIELD_SIZE)

                
            # y座標が下をこえる
            elif ((self.all_agents[i]['p']+self.all_agents[i]['v'])[1] < -FIELD_SIZE):
                
                # ゴール判定
                if (self.all_agents[i]['p'][0] > 
                   self.num_agents_goal[i][self.goal_count[i]][0] - 0.1 
                   and 
                   self.all_agents[i]['p'][0] < 
                   self.num_agents_goal[i][self.goal_count[i]][0] + 0.1 
                   and 
                   self.all_agents[i]['p'][1] > 
                   self.num_agents_goal[i][self.goal_count[i]][1] - 0.1 
                   and 
                   self.all_agents[i]['p'][1] < 
                   self.num_agents_goal[i][self.goal_count[i]][1] + 0.1):
                    
                    # ゴールが調整されているか確認
                    if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                        # 完了時間を算出
                        completion_time = self.calc_completion_time(i, current_step)
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
                        self.goal_temp[i][0] = self.num_agents_goal[i][self.goal_count[i]][0]
                        self.goal_temp[i][1] = self.num_agents_goal[i][self.goal_count[i]][1] + 2*FIELD_SIZE
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
                   self.num_agents_goal[i][self.goal_count[i]][0] - 0.1 
                   and 
                   self.all_agents[i]['p'][0] < 
                   self.num_agents_goal[i][self.goal_count[i]][0] + 0.1 
                   and 
                   self.all_agents[i]['p'][1] > 
                   self.num_agents_goal[i][self.goal_count[i]][1] - 0.1 
                   and 
                   self.all_agents[i]['p'][1] < 
                   self.num_agents_goal[i][self.goal_count[i]][1] + 0.1):
                    
                    # ゴールが調整されているか確認
                    if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                        # 完了時間を算出
                        completion_time = self.calc_completion_time(i, current_step)
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
                        self.goal_temp[i][0] = self.num_agents_goal[i][self.goal_count[i]][0]
                        self.goal_temp[i][1] = self.num_agents_goal[i][self.goal_count[i]][1] + (-2*FIELD_SIZE)
                    else:
                        # はみ出た時用のゴールを初期化
                        self.goal_temp[i][0] = 0
                        self.goal_temp[i][1] = 0
                        

                self.all_agents[i]['p'][1] = -FIELD_SIZE + (
                    (self.all_agents[i]['p']+self.all_agents[i]['v'])[1] - FIELD_SIZE
                )

    
    # 12. 
    def move_agents(self, current_step: int) -> None:
        """
        エージェントを動かす
        更新されるパラメータ：
        self.all_agents
        """
        for i in range(self.num_agents):
            # はみ出た時用のゴールが設定されていない
            # 通常のゴールに向かうベクトルと、回避ベクトルを足したものが速度になる
            if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                self.all_agents[i]['v'] = fs.rotate_vec(
                    np.array([self.goal_vec, 0]), 
                    fs.calc_rad(self.num_agents_goal[i][self.goal_count[i]],
                                self.all_agents[i]['p'])
                )     

            # はみ出た時用のゴールが設定されている
            # はみ出た時用のゴールに向かうベクトルと、回避ベクトルを足したものが速度になる
            else:
                self.all_agents[i]['v'] = fs.rotate_vec(
                    np.array([self.goal_vec, 0]), 
                    fs.calc_rad(self.goal_temp[i], self.all_agents[i]['p'])
                ) 
                
            if self.all_agents[i]['avoidance'] == 'simple': # 単純回避ベクトルを足す
                self.all_agents[i]['v'] += self.simple_avoidance(i)
                
            elif self.all_agents[i]['avoidance'] == 'dynamic': # 動的回避ベクトルを足す
                self.all_agents[i]['v'] += self.dynamic_avoidance(i)
        
        self.check_if_goaled(current_step)
        
        for i in range(self.num_agents):
            # 移動後の座標を確定      
            self.all_agents[i]['p'] += self.all_agents[i]['v']
            
        self.record_parameters(current_step)
    
    
    def simulate(self) -> None:
        """
        self.num_stepsの回数だけエージェントを動かす
        シミュレーションを行うときに最終的に呼び出すメソッド
        """
        for t in tqdm.tqdm(range(self.num_steps)):
            current_step = t + 1
            self.move_agents(current_step)
            row = self.record_agent_information()
            self.data.append(row)
            


    # 14. 
    def plot_positions(self) -> None:
        """
        各エージェントの座標を、エージェントの番号付きでプロットする
        デバッグや新しいメソッドの追加用のメソッド
        プロット中の薄い青色がそのstepでの位置で、濃い青色は次のstepでの位置
        """
        pos_array = self.show_image()
        plt.figure(figsize=(8, 8))
        for i in range(self.num_agents):
            next_pos = self.all_agents[i]['p'] + self.all_agents[i]['v']
            plt.scatter(pos_array[0][i], pos_array[1][i], color='blue', alpha=0.3)
            plt.scatter(*next_pos, color='blue')
            plt.annotate(i, xy=(pos_array[0][i], pos_array[1][i]))
        plt.show()
        
        
    # 15. 
    def approach_detect(self, dist: float) -> np.array: 
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
    
        
    # 16
    def record_agent_information(self) -> np.array:
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
    
        
    # 17
    def record_approaches(self, 
                          approach_dist: Literal['collision','half','quarter','one_eigth'], 
                          step: int, 
                          sim_data: np.array) -> list[int]:
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
            for j in range(step):
                # 一試行で何回エージェントに衝突したか
                total += sim_data[j+1][i]
            
            # 全エージェントの衝突した回数を記録
            approach.append(total)
        
        return approach
        
    def return_results_as_df(self) -> pd.DataFrame:
        """
        1試行の記録をデータフレームにして返す
        """
        # 最後の座標から完了時間を算出
        for i in range(self.num_agents):
            last_completion_time = self.calc_remained_completion_time(i, self.num_steps)
            if not last_completion_time == None:
                self.completion_time.append(last_completion_time)
       
        # 衝突した数、視野の半分、視野の四分の一、視野の八分の一に接近した回数
        collision = self.record_approaches('collision', self.num_steps, self.data)
        half =  self.record_approaches('half', self.num_steps, self.data)
        quarter =  self.record_approaches('quarter', self.num_steps, self.data)
        one_eighth =  self.record_approaches('one_eigth', self.num_steps, self.data)
        
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
                       'simple_avoid_vec': self.simple_avoid_vec}
        
        df_result = pd.DataFrame(dict_result, index=[f'seed_{self.random_seed}'])
        
        return df_result
    
################################################################################        
    def simple_avoidance(self, num: int # エージェントの番号
                          ) -> np.array: # [float, float]
        """
        単純な回避ベクトルの生成(オリジナル)
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
            d = d / (dist_all[num][i] + 2 * self.num_agents_size) # 大きさ1のベクトルにする
            d = d * self.simple_avoid_vec # 大きさを固定値にする
            
            avoid_vec += d # 回避ベクトルを合成する
            
        # ベクトルの平均を出す
        return avoid_vec / len(visible_agents)
    
    
    def dynamic_avoidance(self, num: int) -> np.array:
        """
        動的回避ベクトルの生成(オリジナル)
        ex. self.dynamic_avoidance(num=15) -> array([-0.05, 0.2])
        """
        dist_all = self.calc_distance_all_agents()
        visible_agents = self.find_visible_agents(dist_all, num)
        avoid_vec = np.zeros(2)   # 回避ベクトル
        
        if not visible_agents:
            return avoid_vec    
            
        ### the followings are dynamic vectors ###
        for i in visible_agents:
            # 視野の中心にいるエージェントの位置と速度
            self.num_agents_pos = self.all_agents[num]['p']
            self.num_agents_vel = self.all_agents[num]['v']
            # 視野に入ったエージェントの位置と速度
            self.visible_agent_pos = self.all_agents[i]['p']
            self.visible_agent_vel = self.all_agents[i]['v']

            
            dist_former = dist_all[num][i]
            
            t = 0
            # 2体のエージェントを1ステップ動かして距離を測定
            self.num_agents_pos = self.num_agents_pos + self.num_agents_vel
            self.visible_agent_pos = self.visible_agent_pos + self.visible_agent_vel
            d = self.num_agents_pos - self.visible_agent_pos
            dist_latter = np.linalg.norm(d) - 2 * self.num_agents_size
            
            
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
                    self.num_agents_pos = self.num_agents_pos + self.num_agents_vel
                    self.visible_agent_pos = self.visible_agent_pos + self.visible_agent_vel
                    d = self.num_agents_pos - self.visible_agent_pos
                    dist_latter = np.linalg.norm(d) - 2 * self.num_agents_size
                    
                if dist_former < 0:
                    dcpa = 0 # 最も近い距離で接触している場合はdcpaは0とみなす
                else:
                    dcpa = dist_former * 50 # 単位をピクセルに変換
                    
                tcpa = t

            a1, b1, c1, d1 = abcd['a1'], abcd['b1'], abcd['c1'], abcd['d1']
            # ブレーキ指標の算出
            braking_index = (1 / (1 + np.exp(-c1 - d1 * (tcpa/4000)))) * \
                            (1 / (1 + np.exp(-b1 - a1 * (dcpa/50))))
            
            # dは視界に入ったエージェントに対して反対方向のベクトル
            d = self.all_agents[num]['p'] - self.all_agents[i]['p']
            d = d / (dist_all[num][i] + 2 * self.num_agents_size) # ベクトルの大きさを1にする
            d = d * braking_index # ブレーキ指標の値を反映
            d = d * self.dynamic_avoid_vec # ベクトルの最大値を決定
            
            avoid_vec += d # ベクトルの合成
    
        # ベクトルの平均を出す
        return avoid_vec / len(visible_agents)
