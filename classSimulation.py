""" 
2025/01/07
シミュレーションのためのクラスSimulation

class Simulation
__init__(self)
1. find_goal(self, agent)
2. distance_all_agents(self)
3. approach_detect(self, dist)
4. simple_avoidance(self, num)
5. dynamic_avoidance(self, num)
6. calc_Nic(self, num)
7. find_agents_to_focus(self, num)
8. simulate(self, step)
9. calc_completion_time(self, num, now_step)
10. calc_last_completion_time(self, num)
11. show_image(self)
12. plot_positions(self)
"""

# %% import libraries
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from typing import Literal
import funcSimulation as fs
from funcSimulation import calc_rad, rotate_vec, calc_distance

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

# %% シミュレーションに関わるクラス
class Simulation():
    def __init__(self, 
                 interval: int = 100,
                 agent_size: float = 0.1, 
                 agent: int = 25, 
                 view: int = 1, 
                 viewing_angle: int = 360, 
                 goal_vec: float = 0.06, 
                 avoidance: Literal['simple','dynamic'] = 'simple',
                 simple_avoid_vec: float = 0.06, 
                 dynamic_avoid_vec: float = 0.06):
        
        self.interval = interval # 100msごとにグラフを更新してアニメーションを作成
        self.agent_size = agent_size # エージェントの半径(目盛り) = 5px
        self.agent = agent # エージェント数
        self.view = view # 視野の半径(目盛り) = 50px:エージェント5体分
        self.viewing_angle = viewing_angle # 視野の角度
        self.goal_vec = goal_vec # ゴールベクトルの大きさ(目盛り)
        self.simple_avoid_vec = simple_avoid_vec # 単純回避での回避ベクトルの大きさ(目盛り)
        self.dynamic_avoid_vec = dynamic_avoid_vec # 動的回避での回避ベクトルの最大値(目盛り)
        self.avoidance = avoidance # 'simple' or 'dynamic'

        self.all_agent = [] # 全エージェントの座標を記録
        self.all_agent2 = [] # ゴールの計算用
        self.first_agent = [] # 初期位置記録用
        self.agent_goal = []
        self.first_pos =[]
        
        # エージェントの生成
        for n in range(self.agent):
            # グラフ領域の中からランダムに座標を決定
            pos = np.random.uniform(-FIELD_SIZE, FIELD_SIZE, 2)
            vel = np.random.uniform(-FIELD_SIZE, FIELD_SIZE, 2)
            
            # 座標(0, 0)から座標velへのベクトルがエージェントの初期速度になる
            # self.all_agentの1つの要素に1体のエージェントの位置と速度が記録
            self.all_agent.append(
                {'avoidance': ..., # to caclculate the awareness but may not be useful
                 'p': pos, 
                 'v': rotate_vec(np.array([self.goal_vec, 0]), 
                                 calc_rad(vel, np.array([0, 0])))
                 }
            )
            
        # 初期位置と初期速度をコピー
        self.all_agent2 = deepcopy(self.all_agent)
        self.first_agent = deepcopy(self.all_agent)
        
        # エージェントの初期位置を保存
        for i in range(self.agent):
            self.first_pos.append(self.first_agent[i]['p'])
        
        # エージェントにゴールを8ずつ設定
        for i in range(self.agent):
            goals = []
            for j in range(8):
                goals.append(self.find_goal(self.all_agent2[i]))
                
            self.agent_goal.append(goals)
             
            
        # エージェント間の距離を記録するリスト
        self.dist = np.zeros([self.agent, self.agent])
        
        # ゴールした回数を記録するリスト
        self.goal_count = []
        for i in range(self.agent):
            self.goal_count.append(0)

        # はみ出た時用のゴール
        self.goal_temp = np.zeros([self.agent, 2])
        
        # 完了時間を測るための変数
        self.start_step = np.zeros([self.agent])
        self.goal_step = np.zeros([self.agent])
        self.start_pos = self.first_pos
        self.goal_pos = np.zeros([self.agent, 2])
         
        # 完了時間を記録するリスト
        self.completion_time = []


    # 1. 
    def find_goal(self, agent: dict[str, np.array] 
                  ) -> np.array: # [float, float]
        """ 
        ゴールの計算
        エージェントが初期速度のまま進んだ場合に通過する、グラフ領域の境界線の座標
        初期位置の値を使うためself.all_agentではなくself.all_agent2を使う
        ex. self.find_goal(agent=self.all_agent2[10]) -> array([-3.0, -5.0])
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
                    print("調整入りました")
                    
                # y座標も同時にサイズを超えるかつyが負
                elif (goal[1] < -FIELD_SIZE + 0.1):
                    # ゴールの座標がグラフ領域の角にならないように調整
                    goal[1] = goal[1] + 0.1
                    print("調整入りました")
                    
                goal[0] = -FIELD_SIZE
                # 端に到達したエージェントを、反対側の端に移動させる
                agent['p'][0] = FIELD_SIZE + ((agent['p'] + agent['v'])[0] + FIELD_SIZE)
                break
                        
            elif ((agent['p'] + agent['v'])[0] > FIELD_SIZE):
                goal = agent['p'] + agent['v']
                
                # y座標も同時にサイズを超えるかつyが正
                if (goal[1] > FIELD_SIZE - 0.1):
                    goal[1] = goal[1] - 0.1
                    print("調整入りました")

               # y座標も同時にサイズを超えるかつyが負
                elif (goal[1] < -FIELD_SIZE + 0.1):
                    goal[1] = goal[1] + 0.1
                    print("調整入りました")
                    
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
    def distance_all_agents(self):
        """ 
        全エージェントについて、その他のエージェントとの距離を計算し、self.distに格納
        """
        for i in range(self.agent):
            for j in range(self.agent):
                d = self.all_agent[i]['p'] - self.all_agent[j]['p']
                # エージェント間の距離を算出、エージェントのサイズも考慮
                self.dist[i][j] = np.linalg.norm(d) - 2 * self.agent_size
                
                
    # 3. 
    def approach_detect(self, dist: float) -> np.array: 
        """ 
        指定した距離より接近したエージェントの数を返す
        ex. self.approach_detect(dist=0.5) -> array([[0, 3],[1, 2],...[24, 1]])
        """
        self.distance_all_agents()
        approach_agent = []
        
        # それぞれのエージェントについて、distより接近したエージェントの数を記録
        for t in range(self.agent):
            visible_agents = [i for i, x in enumerate(self.dist[t]) 
                              if x != -(0.2) and x < dist]
            approach_agent.append([t, len(visible_agents)])
        approach_agent = np.array(approach_agent) 
            
        return approach_agent
    
    
    # 4. 
    def simple_avoidance_with_focus(self, num: int # エージェントの番号
                                    ) -> np.array: # [float, float]
        """
        単純な回避ベクトルの生成(awareness modelで計算相手を選定)
        ex. self.simple_avoidance(num=15) -> array([-0.05, 0.02])
        """
        self.distance_all_agents()
        # near_agentsは360度の視野に入ったエージェント
        # visible_agentsは視野を狭めた場合に視野に入ったエージェント
        near_agents = [i for i, x in enumerate(self.dist[num]) 
                       if x != -(0.2) and x < self.view]
        visible_agents = []
        
        # 回避ベクトル
        avoid_vec = np.zeros(2)
        
        if not near_agents:
            return avoid_vec
        
        # near_agentsについて、awareness modelの値を基にしたエージェントを計算対象にする
        # focus_agents = self.calc_Nic(num)[near_agents]
        
        # ゴールベクトルの角度を算出する
        goal_angle = np.degrees(
            calc_rad(self.agent_goal[num][self.goal_count[num]], 
                     self.all_agent[num]['p'])
        )

        for i in near_agents:
            # 近づいたエージェントとの角度を算出
            agent_angle = np.degrees(
                calc_rad(self.all_agent[i]['p'], 
                         self.all_agent[num]['p'])
            )
            
            # 近づいたエージェントとの角度とゴールベクトルの角度の差を計算
            angle_difference = abs(goal_angle - agent_angle)
            
            if angle_difference > 180:
                angle_difference = 360 - angle_difference
                
            # 視野に入っているエージェントをvisible_agentsに追加
            if angle_difference <= self.viewing_angle / 2:
                visible_agents.append(i)
                
        if not visible_agents: 
            return avoid_vec
            
        for i in visible_agents:
            # dは視界に入ったエージェントに対して反対方向のベクトル
            d = self.all_agent[num]['p'] - self.all_agent[i]['p']
            d = d / (self.dist[num][i] + 2 * self.agent_size) # 大きさ1のベクトルにする
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
        self.distance_all_agents()
        near_agents = [i for i, x in enumerate(self.dist[num]) 
                       if x != -(0.2) and x < self.view]
        visible_agents = []
        avoid_vec = np.zeros(2)
        
        if not near_agents:
            return avoid_vec
        
        # near_agentsについて、awareness modelの値を基にしたエージェントを計算対象にする
        # focus_agents = self.calc_Nic(num)[near_agents]
        
        # ゴールベクトルの角度を算出する
        goal_angle = np.degrees(
            calc_rad(self.agent_goal[num][self.goal_count[num]], 
                     self.all_agent[num]['p'])
        )

        for i in near_agents:
            # 近づいたエージェントとの角度を算出
            agent_angle = np.degrees(
                calc_rad(self.all_agent[i]['p'], 
                         self.all_agent[num]['p'])
            )
            
            # 近づいたエージェントとの角度とゴールベクトルの角度の差を計算
            angle_difference = abs(goal_angle - agent_angle)

            if angle_difference > 180:
                angle_difference = 360 - angle_difference
            
            # 視界に入ったエージェントをvisible_agentsに追加
            if angle_difference <= self.viewing_angle / 2:
                visible_agents.append(i)
                
        if not visible_agents:
            return avoid_vec
            
        
        for i in visible_agents:
            # 視野の中心にいるエージェントの位置と速度
            self.agent_pos = self.all_agent[num]['p']
            self.agent_vel = self.all_agent[num]['v']
            # 視野に入ったエージェントの位置と速度
            self.visible_agent_pos = self.all_agent[i]['p']
            self.visible_agent_vel = self.all_agent[i]['v']

            
            dist_former = self.dist[num][i]
            
            t = 0
            # 2体のエージェントを1ステップ動かして距離を測定
            self.agent_pos = self.agent_pos + self.agent_vel
            self.visible_agent_pos = self.visible_agent_pos + self.visible_agent_vel
            d = self.agent_pos - self.visible_agent_pos
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
                    self.agent_pos = self.agent_pos + self.agent_vel
                    self.visible_agent_pos = self.visible_agent_pos + self.visible_agent_vel
                    d = self.agent_pos - self.visible_agent_pos
                    dist_latter = np.linalg.norm(d) - 2 * self.agent_size
                    
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
            d = self.all_agent[num]['p'] - self.all_agent[i]['p']
            d = d / (self.dist[num][i] + 2 * self.agent_size) # ベクトルの大きさを1にする
            d = d * braking_index # ブレーキ指標の値を反映
            d = d * self.dynamic_avoid_vec # ベクトルの最大値を決定
            
            avoid_vec += d # ベクトルの合成
    
        # ベクトルの平均を出す
        return avoid_vec / len(visible_agents)
    
    
    # 6. 
    def calc_Nic(self, num: int) -> np.array:
        """ 
        numで指定したエージェントについて、Nic(Number in the circle)を計算する
        1. エージェントAとエージェントBの中点cpを計算し、cpから他の全てのエージェントXとの距離cpxを計算
        2. 1の中で、cpとエージェントAの距離dist_cp_meより小さいcpxの数を計算
        ex. self.calc_Nic(num=10) -> array([[0,1], [1,2],...,[24,12]]) (相手,Nic)
        """ 
        all_Nics = []
        posA = self.all_agent[num]['p']
        for i in range(self.agent):
            posB = self.all_agent[i]['p']
            cp = ( (posA[0]+posB[0])/2, (posA[1]+posB[1])/2 )
            radius = calc_distance(*cp, *posA)
            
            Nic_agents = []
            for j in range(self.agent):
                posX = self.all_agent[j]['p']
                dist_cp_posX = calc_distance(*posX, *cp)
                if dist_cp_posX <= radius:
                    Nic_agents.append(i)
            all_Nics.append([i, len(Nic_agents)-2]) # posAとposBの分を引く
        all_Nics = np.array(all_Nics)
        
        return all_Nics
    
    
    # 7. 
    def find_agents_to_focus(self, num: int) -> np.array:
        """
        awareness modelを用いて、交流相手を選定する
        ex. self.find_agents_to_focus(
            num=10) -> array([[0, 0.2], [1, 0.3],...,[24, 0.8]]) (相手,awareness)
        """ 
        agents_to_focus = []
        my_posx, my_posy = self.all_agent[num]['p']
        vselfxy = self.all_agent[num]['v']
        Vself = np.linalg.norm(vselfxy)
        
        a = self.all_agent[num]['p'] + self.all_agent[num]['v']
        b = self.all_agent[num]['p']
        line1 = np.array([b, a])
        
        for i in range(self.agent):
            other_posx, other_posy = self.all_agent[i]['p']
            Px = other_posx - my_posx
            Py = other_posy - my_posy
            votherxy = self.all_agent[i]['v']
            Vother = np.linalg.norm(votherxy)
            
            c = self.all_agent[i]['p']
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
    def simulate(self, now_step: int) -> None:
        """
        エージェントを動かす
        """
        for i in range(self.agent):
            # はみ出た時用のゴールが設定されていない
            # 通常のゴールに向かうベクトルと、回避ベクトルを足したものが速度になる
            if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                self.all_agent[i]['v'] = rotate_vec(
                    np.array([self.goal_vec, 0]), 
                    calc_rad(self.agent_goal[i][self.goal_count[i]],
                             self.all_agent[i]['p'])
                )     

            # はみ出た時用のゴールが設定されている
            # はみ出た時用のゴールに向かうベクトルと、回避ベクトルを足したものが速度になる
            else:
                self.all_agent[i]['v'] = rotate_vec(
                    np.array([self.goal_vec, 0]), 
                    calc_rad(self.goal_temp[i], self.all_agent[i]['p'])
                ) 
                
            if self.avoidance == 'simple': # 単純回避ベクトルを足す
                self.all_agent[i]['v'] += self.simple_avoidance(i)
                
            elif self.avoidance == 'dynamic': # 動的回避ベクトルを足す
                self.all_agent[i]['v'] += self.dynamic_avoidance(i)
        
        
        for i in range(self.agent):
            # x座標が左端をこえる
            if ((self.all_agent[i]['p'] + self.all_agent[i]['v'])[0] < -FIELD_SIZE):
                # ゴールに到着
                if (self.all_agent[i]['p'][0] > 
                   self.agent_goal[i][self.goal_count[i]][0] - 0.1 
                   and 
                   self.all_agent[i]['p'][0] < 
                   self.agent_goal[i][self.goal_count[i]][0] + 0.1
                   and 
                   self.all_agent[i]['p'][1] > 
                   self.agent_goal[i][self.goal_count[i]][1] - 0.1 
                   and 
                   self.all_agent[i]['p'][1] < 
                   self.agent_goal[i][self.goal_count[i]][1] + 0.1):
                    
                    # 通常のゴールに到着
                    if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                        # 完了時間を算出
                        completion_time = self.calc_completion_time(i, now_step)
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
                        self.goal_temp[i][0] = self.agent_goal[i][self.goal_count[i]][0] + 2*FIELD_SIZE
                        self.goal_temp[i][1] = self.agent_goal[i][self.goal_count[i]][1]
                        
                    # はみ出た時用のゴールが設定されている
                    else:
                        # はみ出た時用のゴールを初期化
                        self.goal_temp[i][0] = 0
                        self.goal_temp[i][1] = 0
                
                # エージェントを反対の端へ移動
                self.all_agent[i]['p'][0] = FIELD_SIZE + (
                    (self.all_agent[i]['p']+self.all_agent[i]['v'])[0] + FIELD_SIZE
                )
        
            
            # x座標が右端をこえる
            elif ((self.all_agent[i]['p']+self.all_agent[i]['v'])[0] > FIELD_SIZE):
                
                # ゴール判定
                if (self.all_agent[i]['p'][0] > 
                   self.agent_goal[i][self.goal_count[i]][0] - 0.1 
                   and 
                   self.all_agent[i]['p'][0] < 
                   self.agent_goal[i][self.goal_count[i]][0] + 0.1 
                   and 
                   self.all_agent[i]['p'][1] > 
                   self.agent_goal[i][self.goal_count[i]][1] - 0.1 
                   and 
                   self.all_agent[i]['p'][1] < 
                   self.agent_goal[i][self.goal_count[i]][1] + 0.1):
                    
                    # ゴールが調整されているか確認
                    if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                        # 完了時間を算出
                        completion_time = self.calc_completion_time(i, now_step)
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
                        self.goal_temp[i][0] = self.agent_goal[i][self.goal_count[i]][0] + (-2 * FIELD_SIZE)
                        self.goal_temp[i][1] = self.agent_goal[i][self.goal_count[i]][1]
                    else:
                        # はみ出た時用のゴールを初期化
                        self.goal_temp[i][0] = 0
                        self.goal_temp[i][1] = 0
                        
                self.all_agent[i]['p'][0] = -FIELD_SIZE + \
                    ((self.all_agent[i]['p']+self.all_agent[i]['v'])[0] - FIELD_SIZE)

                
            # y座標が下をこえる
            elif ((self.all_agent[i]['p']+self.all_agent[i]['v'])[1] < -FIELD_SIZE):
                
                # ゴール判定
                if (self.all_agent[i]['p'][0] > 
                   self.agent_goal[i][self.goal_count[i]][0] - 0.1 
                   and 
                   self.all_agent[i]['p'][0] < 
                   self.agent_goal[i][self.goal_count[i]][0] + 0.1 
                   and 
                   self.all_agent[i]['p'][1] > 
                   self.agent_goal[i][self.goal_count[i]][1] - 0.1 
                   and 
                   self.all_agent[i]['p'][1] < 
                   self.agent_goal[i][self.goal_count[i]][1] + 0.1):
                    
                    # ゴールが調整されているか確認
                    if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                        # 完了時間を算出
                        completion_time = self.calc_completion_time(i, now_step)
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
                        self.goal_temp[i][0] = self.agent_goal[i][self.goal_count[i]][0]
                        self.goal_temp[i][1] = self.agent_goal[i][self.goal_count[i]][1] + 2*FIELD_SIZE
                    else:        
                        # はみ出た時用のゴールを初期化
                        self.goal_temp[i][0] = 0
                        self.goal_temp[i][1] = 0
                        
                self.all_agent[i]['p'][1] = FIELD_SIZE + (
                    (self.all_agent[i]['p']+self.all_agent[i]['v'])[1] + FIELD_SIZE
                )
                
            # y座標が上をこえる     
            elif ((self.all_agent[i]['p']+self.all_agent[i]['v'])[1] > FIELD_SIZE):
                
                # ゴール判定
                if (self.all_agent[i]['p'][0] > 
                   self.agent_goal[i][self.goal_count[i]][0] - 0.1 
                   and 
                   self.all_agent[i]['p'][0] < 
                   self.agent_goal[i][self.goal_count[i]][0] + 0.1 
                   and 
                   self.all_agent[i]['p'][1] > 
                   self.agent_goal[i][self.goal_count[i]][1] - 0.1 
                   and 
                   self.all_agent[i]['p'][1] < 
                   self.agent_goal[i][self.goal_count[i]][1] + 0.1):
                    
                    # ゴールが調整されているか確認
                    if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                        # 完了時間を算出
                        completion_time = self.calc_completion_time(i, now_step)
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
                        self.goal_temp[i][0] = self.agent_goal[i][self.goal_count[i]][0]
                        self.goal_temp[i][1] = self.agent_goal[i][self.goal_count[i]][1] + (-2*FIELD_SIZE)
                    else:
                        # はみ出た時用のゴールを初期化
                        self.goal_temp[i][0] = 0
                        self.goal_temp[i][1] = 0
                        

                self.all_agent[i]['p'][1] = -FIELD_SIZE + (
                    (self.all_agent[i]['p']+self.all_agent[i]['v'])[1] - FIELD_SIZE
                )

                
        for i in range(self.agent):
            # 移動後の座標を確定      
            self.all_agent[i]['p'] = self.all_agent[i]['p'] + self.all_agent[i]['v']
            
         
    # 9. 
    def calc_completion_time(self, num: int, now_step: int) -> float:
        """ 
        完了時間を記録
        ex. self.calc_completion_time(num=10, now_step=10) -> 35.6
        """
        # 一回のゴールにおける初めのステップと終わりのステップを記録
        self.start_step[num] = self.goal_step[num] + 1
        self.goal_step[num] = now_step
        
        # 一回目のゴール
        if (self.start_step[num] == 1):
            # 一回目のゴールにおける、ゴール位置を記録
            self.goal_pos[num] = self.agent_goal[num][self.goal_count[num]]
            
        # 一回目以降のゴール
        else:
            # 前回のゴールが左端にあるとき
            if (self.goal_pos[num][0] == -FIELD_SIZE):
                # スタート位置、ゴール位置を記録
                self.start_pos[num][0] = self.goal_pos[num][0] + 2*FIELD_SIZE
                self.start_pos[num][1] = self.goal_pos[num][1]
                self.goal_pos[num] = self.agent_goal[num][self.goal_count[num]]
                
            # 前回のゴールが右端にあるとき
            elif (self.goal_pos[num][0] == FIELD_SIZE):
                # スタート位置、ゴール位置を記録
                self.start_pos[num][0] = self.goal_pos[num][0] + (-2*FIELD_SIZE)
                self.start_pos[num][1] = self.goal_pos[num][1]
                self.goal_pos[num] = self.agent_goal[num][self.goal_count[num]]
            
            # 前回のゴールが下端にあるとき
            elif (self.goal_pos[num][1] == -FIELD_SIZE):
                # スタート位置、ゴール位置を記録
                self.start_pos[num][0] = self.goal_pos[num][0]
                self.start_pos[num][1] = self.goal_pos[num][1] + 2*FIELD_SIZE
                self.goal_pos[num] = self.agent_goal[num][self.goal_count[num]]
                
            # 前回のゴールが上端にあるとき
            elif (self.goal_pos[num][1] == FIELD_SIZE):
                # スタート位置、ゴール位置を記録
                self.start_pos[num][0] = self.goal_pos[num][0]
                self.start_pos[num][1] = self.goal_pos[num][1] + (-2*FIELD_SIZE)
                self.goal_pos[num] = self.agent_goal[num][self.goal_count[num]]
                
        # 完了時間を計算(ゴールまでのステップ/ゴールまでの距離)
        completion_time = (self.goal_step[num] - self.start_step[num] + 1) / \
                          (np.linalg.norm(self.start_pos[num] - self.goal_pos[num]))
        
        # 外れ値を除外
        if (completion_time > 200):
            print("消します")
            print(completion_time)
            return None
        
        return completion_time
    
    
    # 10. 
    def calc_last_completion_time(self, num: int, step: int) -> float:
        """
        最後の座標から完了時間を算出(やらなくても良いかもしれない)
        """
        # 初めのステップと終わりのステップを記録
        self.start_step[num] = self.goal_step[num] + 1
        self.goal_step[num] = step
        
        # ゴールする前に境界をはみ出ている場合
        if not (self.goal_temp[num][0] == 0 and self.goal_temp[num][1] == 0):
            # 左右の境界をはみ出た
            if (abs(self.all_agent[num]['p'][0]) > abs(self.all_agent[num]['p'][1])):
                # はみ出る前に戻してあげる
                self.all_agent[num]['p'][0] = -self.all_agent[num]['p'][0]
            # 上下の境界をはみ出た
            elif (abs(self.all_agent[num]['p'][0]) < abs(self.all_agent[num]['p'][1])):
                # はみ出る前に戻してあげる
                self.all_agent[num]['p'][1] = -self.all_agent[num]['p'][1]
            
        # スタート位置、ゴール位置を算出
        # 前回のゴールが左端にあるとき
        if (self.goal_pos[num][0] == -FIELD_SIZE):
            self.start_pos[num][0] = self.goal_pos[num][0] + 2 * FIELD_SIZE
            self.start_pos[num][1] = self.goal_pos[num][1]
            self.goal_pos[num] = self.agent_goal[num][self.goal_count[num]]

        # 前回のゴールが右端にあるとき
        elif (self.goal_pos[num][0] == FIELD_SIZE):
            self.start_pos[num][0] = self.goal_pos[num][0] + (-2 * FIELD_SIZE)
            self.start_pos[num][1] = self.goal_pos[num][1]
            self.goal_pos[num] = self.agent_goal[num][self.goal_count[num]]

        # 前回のゴールが下端にあるとき
        elif (self.goal_pos[num][1] == -FIELD_SIZE):
            self.start_pos[num][0] = self.goal_pos[num][0]
            self.start_pos[num][1] = self.goal_pos[num][1] + 2 * FIELD_SIZE
            self.goal_pos[num] = self.agent_goal[num][self.goal_count[num]]

        # 前回のゴールが上端にあるとき
        elif (self.goal_pos[num][1] == FIELD_SIZE):
            self.start_pos[num][0] = self.goal_pos[num][0]
            self.start_pos[num][1] = self.goal_pos[num][1] + (-2 * FIELD_SIZE)
            self.goal_pos[num] = self.agent_goal[num][self.goal_count[num]]
            
        
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
        d = -(c * self.all_agent[num]['p'][0]) + self.all_agent[num]['p'][1]

        # 2つの直線の交点を算出
        cross_x = (b - d) / (-(a - c))
        cross_y = a * cross_x + b
        cross = np.array([cross_x, cross_y])
        
        # スタートから交点までの距離を計算
        distance = np.linalg.norm(self.start_pos[num] - cross)
        
        # 完了時間を計算(ゴールまでのステップ/ゴールまでの距離)
        completion_time = (self.goal_step[num] - self.start_step[num] + 1) / distance

        if (completion_time > 200 or completion_time < 10):
            print("消しました")
            print(completion_time)
            return None
        
        return completion_time


    # 11. 
    def show_image(self) -> np.array: # shape(2(x,y), エージェントの数)
        """
        プロットのための座標を送る
        ex. self.show_image() -> array([[2.0, 1.2,...,3.5],
                                       [2.5, 1.5,...,1.5]]) (2, エージェントの数)
        """
        pos_array = np.zeros([2, self.agent])
        for i in range(self.agent):
            pos_array[0][i] = self.all_agent[i]['p'][0]
            pos_array[1][i] = self.all_agent[i]['p'][1]
            
        return pos_array


    # 12. 
    def plot_positions(self) -> None:
        """
        各エージェントの座標を、エージェントの番号付きでプロットする
        薄い青色がそのstepでの位置で、濃い青色は次のstepでの位置
        """
        pos_array = self.show_image()
        plt.figure(figsize=(8, 8))
        for i in range(self.agent):
            next_pos = self.all_agent[i]['p'] + self.all_agent[i]['v']
            plt.scatter(pos_array[0][i], pos_array[1][i], color='blue', alpha=0.3)
            plt.scatter(*next_pos, color='blue')
            plt.annotate(i, xy=(pos_array[0][i], pos_array[1][i]))
        plt.show()
        
    # 13
    def record_agent_information(self) -> np.array:
        """
        全エージェントの位置と速度、接近を記録
        """
        # 初期の位置と速度を記録
        row = np.concatenate([self.all_agent[0]['p'], self.all_agent[0]['v']])
        # rowにはある時刻の全エージェントの位置と速度が入る
        for i in range(1, self.agent):
            row = np.concatenate([row, self.all_agent[i]['p'], self.all_agent[i]['v']])
            
        # エージェントの接近を記録
        # 衝突したエージェント
        collision_agent = self.approach_detect(0)
        for i in range(self.agent):
            row = np.append(row, collision_agent[i][1])

        # 視野の半分まで接近したエージェント
        collision_agent = self.approach_detect(0.5)
        for i in range(self.agent):
            row = np.append(row, collision_agent[i][1])
            
        # 視野の4分の1まで接近したエージェント
        collision_agent = self.approach_detect(0.25)
        for i in range(self.agent):
            row = np.append(row, collision_agent[i][1])

        # 視野の8分の1まで接近したエージェント
        collision_agent = self.approach_detect(0.125)
        for i in range(self.agent):
            row = np.append(row, collision_agent[i][1])
            
        return row
        
    # 14
    def record_approaches(self, 
                          approach_dist: Literal['collision','half','quarter','one_eigth'], 
                          step: int, 
                          sim_data: np.array) -> list[int]:
        """
        エージェント同士が接近した回数を、接近度合い別で出力する
        ex. self.record_approaches('collision', STEP=500, data) -> [0,0,3,...,12]
        """
        if approach_dist == 'collision':
            start, stop = 4*self.agent, 5*self.agent
            
        elif approach_dist == 'half':
            start, stop = 5*self.agent, 6*self.agent
            
        elif approach_dist == 'quarter':
            start, stop = 6*self.agent, 7*self.agent
            
        elif approach_dist == 'one_eigth':
            start, stop = 7*self.agent, 8*self.agent
            
        approach = []
        for i in range(start, stop, 1):
            total = 0
            for j in range(step):
                # 一試行で何回エージェントに衝突したか
                total += sim_data[j+1][i]
            
            # 全エージェントの衝突した回数を記録
            approach.append(total)
        
        return approach
        
################################################################################        
    def simple_avoidance(self, num: int # エージェントの番号
                          ) -> np.array: # [float, float]
        """
        単純な回避ベクトルの生成(オリジナル)
        ex. self.simple_avoidance(num=15) -> array([-0.05, 0.02])
        """
        self.distance_all_agents()
        # near_agentsは360度の視野に入ったエージェント
        # visible_agentsは視野を狭めた場合に視野に入ったエージェント
        near_agents = [i for i, x in enumerate(self.dist[num]) 
                        if x != -(0.2) and x < self.view]
        visible_agents = []
        
        # 回避ベクトル
        avoid_vec = np.zeros(2)
        
        if not near_agents:
            return avoid_vec
        
        # ゴールベクトルの角度を算出する
        goal_angle = np.degrees(
            calc_rad(self.agent_goal[num][self.goal_count[num]], 
                      self.all_agent[num]['p'])
        )

        for i in near_agents:
            # 近づいたエージェントとの角度を算出
            agent_angle = np.degrees(
                calc_rad(self.all_agent[i]['p'], 
                          self.all_agent[num]['p'])
            )
            
            # 近づいたエージェントとの角度とゴールベクトルの角度の差を計算
            angle_difference = abs(goal_angle - agent_angle)
            
            if angle_difference > 180:
                angle_difference = 360 - angle_difference
                
            # 視野に入っているエージェントをvisible_agentsに追加
            if angle_difference <= self.viewing_angle / 2:
                visible_agents.append(i)
                
        if not visible_agents: 
            return avoid_vec
            
        for i in visible_agents:
            # dは視界に入ったエージェントに対して反対方向のベクトル
            d = self.all_agent[num]['p'] - self.all_agent[i]['p']
            d = d / (self.dist[num][i] + 2 * self.agent_size) # 大きさ1のベクトルにする
            d = d * self.simple_avoid_vec # 大きさを固定値にする
            
            avoid_vec += d # 回避ベクトルを合成する
            
        # ベクトルの平均を出す
        return avoid_vec / len(visible_agents)
    
    
    def dynamic_avoidance(self, num: int) -> np.array:
        """
        動的回避ベクトルの生成(オリジナル)
        ex. self.dynamic_avoidance(num=15) -> array([-0.05, 0.2])
        """
        self.distance_all_agents()
        near_agents = [i for i, x in enumerate(self.dist[num]) 
                        if x != -(0.2) and x < self.view]
        visible_agents = []
        avoid_vec = np.zeros(2)
        
        if not near_agents:
            return avoid_vec
        
        # ゴールベクトルの角度を算出する
        goal_angle = np.degrees(
            calc_rad(self.agent_goal[num][self.goal_count[num]], 
                     self.all_agent[num]['p'])
        )

        for i in near_agents:
            # 近づいたエージェントとの角度を算出
            agent_angle = np.degrees(
                calc_rad(self.all_agent[i]['p'], 
                         self.all_agent[num]['p'])
            )
            
            # 近づいたエージェントとの角度とゴールベクトルの角度の差を計算
            angle_difference = abs(goal_angle - agent_angle)

            if angle_difference > 180:
                angle_difference = 360 - angle_difference
            
            # 視界に入ったエージェントをvisible_agentsに追加
            if angle_difference <= self.viewing_angle / 2:
                visible_agents.append(i)
                
        if not visible_agents:
            return avoid_vec
            
        
        for i in visible_agents:
            # 視野の中心にいるエージェントの位置と速度
            self.agent_pos = self.all_agent[num]['p']
            self.agent_vel = self.all_agent[num]['v']
            # 視野に入ったエージェントの位置と速度
            self.visible_agent_pos = self.all_agent[i]['p']
            self.visible_agent_vel = self.all_agent[i]['v']

            
            dist_former = self.dist[num][i]
            
            t = 0
            # 2体のエージェントを1ステップ動かして距離を測定
            self.agent_pos = self.agent_pos + self.agent_vel
            self.visible_agent_pos = self.visible_agent_pos + self.visible_agent_vel
            d = self.agent_pos - self.visible_agent_pos
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
                    self.agent_pos = self.agent_pos + self.agent_vel
                    self.visible_agent_pos = self.visible_agent_pos + self.visible_agent_vel
                    d = self.agent_pos - self.visible_agent_pos
                    dist_latter = np.linalg.norm(d) - 2 * self.agent_size
                    
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
            d = self.all_agent[num]['p'] - self.all_agent[i]['p']
            d = d / (self.dist[num][i] + 2 * self.agent_size) # ベクトルの大きさを1にする
            d = d * braking_index # ブレーキ指標の値を反映
            d = d * self.dynamic_avoid_vec # ベクトルの最大値を決定
            
            avoid_vec += d # ベクトルの合成
    
        # ベクトルの平均を出す
        return avoid_vec / len(visible_agents)
        