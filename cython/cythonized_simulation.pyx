# cython: language_level = 3
# cython: boundcheck=False
# cython: wraparound=False
from copy import deepcopy

import cython
from cython import boundscheck, wraparound
import numpy as np
cimport numpy as cnp
cnp.import_array()

from libc.math cimport sin, cos, atan2, M_PI
from libc.math cimport round as c_round
from libc.math cimport sqrt as c_sqrt
from libc.math cimport exp as c_exp

ctypedef cnp.float64_t DTYPE_t

cdef:
    int INTERVAL = 100
    int VIEWING_ANGLE = 360
    int VIEW = 1
    int FIELD_SIZE = 5
    DTYPE_t DYNAMIC_AVOID_VEC = 0.06
    DTYPE_t GOAL_VEC = 0.06
    DTYPE_t AGENT_SIZE = 0.1
    DTYPE_t PI = M_PI

# できるだけnumpy関数は使わずに配列の作成のみに使うようにする
# 配列へのアクセスはインデックスアクセスをするようにする

cdef DTYPE_t calc_rad (DTYPE_t[:] pos2, 
                       DTYPE_t[:] pos1): 
    
    return atan2(pos2[1]-pos1[1], pos2[0]-pos1[0])
    

cdef DTYPE_t[:] rotate_vec (DTYPE_t[:] vec, 
                            DTYPE_t rad):

    cdef DTYPE_t sin_val, cos_val
    cdef DTYPE_t[:] retval

    # Use libc math functions for sin and cos
    sin_val = sin(rad)
    cos_val = cos(rad)
    
    # Perform matrix multiplication manually (since it's a 2x2 matrix)
    retval = vec.copy()  # Create a copy to store the rotated result
    retval[0] = cos_val * vec[0] - sin_val * vec[1]  # Rotation for X-axis
    retval[1] = sin_val * vec[0] + cos_val * vec[1]  # Rotation for Y-axis

    return retval


cdef DTYPE_t to_deg (DTYPE_t rad):
    return rad * (180. / PI)


cdef DTYPE_t c_norm (DTYPE_t dx,
                     DTYPE_t dy):
    return  c_sqrt(dx * dx + dy * dy) 


cdef DTYPE_t[:,:,:] generate_agents (
    int num_agents,
    DTYPE_t dynamic_percent,
    int random_seed
):
    cdef:
        DTYPE_t[:,:,:] all_agents
        #DTYPE_t[:] zero_arr
        DTYPE_t[:] goal_arr
        int num_dynamic_agent
        int n
        int avoidance
        DTYPE_t[:] pos, vel,             
        DTYPE_t[:] rot_v

    # 動的回避を行うエージェントの数
    num_dynamic_agent = int(c_round(num_agents*dynamic_percent))
    
    all_agents = np.zeros([num_agents, 3, 2]) # 全エージェントの座標を記録
    # エージェントの生成
    goal_arr = np.array([GOAL_VEC, 0.])
    zero_arr = np.zeros(2)
    
    np.random.seed(random_seed)
    for n in range(num_agents):
        # グラフ領域の中からランダムに座標を決定
        pos = np.random.uniform(-FIELD_SIZE, FIELD_SIZE, 2)
        vel = np.random.uniform(-FIELD_SIZE, FIELD_SIZE, 2)
        rot_v = rotate_vec(goal_arr, calc_rad(vel, zero_arr))
        
        # 1は動的回避で、0は単純回避
        if n < num_dynamic_agent:
            avoidance = 1
        else:
            avoidance = 0
        
        # 座標(0, 0)から座標velへのベクトルがエージェントの初期速度になる
        # self.all_agentsの1つの要素に1体のエージェントの位置と速度が記録
        # P(t) - V(t) = P(t-1)
        all_agents[n, 0] = avoidance
        all_agents[n, 1, 0] = pos[0]
        all_agents[n, 1, 1] = pos[1]
        all_agents[n, 2, 0] = rot_v[0]
        all_agents[n, 2, 1] = rot_v[1]

    return all_agents


cdef DTYPE_t[:,:] save_init_pos (
    DTYPE_t[:,:,:] all_agents,
    int num_agents
):
    cdef:
        DTYPE_t[:,:] first_pos = np.empty([num_agents, 2], dtype=np.float64)
        cnp.ndarray[DTYPE_t, ndim=3] first_agent
        int i
    
    # エージェントの初期位置を保存
    first_agent = deepcopy(np.asarray(all_agents))
    for i in range(num_agents):
        first_pos[i, 0] = first_agent[i, 1, 0]
        first_pos[i, 1] = first_agent[i, 1, 1]  

    return first_pos


cdef DTYPE_t[:] calc_goals ( 
    DTYPE_t[:] agent_pos,
    DTYPE_t[:] agent_vel
): 
    """ 
    ゴールのxy座標の計算
    エージェントが初期速度のまま進んだ場合に通過する、グラフ領域の境界線の座標
    初期位置の値を使うためself.all_agentsではなくself.all_agents2を使う
    ex. self.set_goals(agent=self.all_agents2[10]) -> array([-3.0, -5.0])
    """
    cdef:
        DTYPE_t[:] goal = np.empty(2, dtype=np.float64)
        
    while True:
        goal[0] = agent_pos[0] + agent_vel[0]
        goal[1] = agent_pos[1] + agent_vel[1]
        
        # x座標がグラフ領域を超える
        if (agent_pos[0] + agent_vel[0]) < -FIELD_SIZE:
            # 超えた時の座標をゴールとする
            
            # y座標も同時にサイズを超えるかつyが正
            if (goal[1] > FIELD_SIZE - 0.1):
                # ゴールの座標がグラフ領域の角にならないように調整
                goal[1] -= 0.1
                
            # y座標も同時にサイズを超えるかつyが負
            elif (goal[1] < -FIELD_SIZE + 0.1):
                # ゴールの座標がグラフ領域の角にならないように調整
                goal[1] += 0.1
                
            goal[0] = -FIELD_SIZE
            # 端に到達したエージェントを、反対側の端に移動させる
            agent_pos[0] = 2*FIELD_SIZE + (agent_pos[0] + agent_vel[0])
            break
                    
        
        elif (agent_pos[0] + agent_vel[0]) > FIELD_SIZE:         
            # y座標も同時にサイズを超えるかつyが正
            if (goal[1] > FIELD_SIZE - 0.1):
                goal[1] -= 0.1

            # y座標も同時にサイズを超えるかつyが負
            elif (goal[1] < -FIELD_SIZE + 0.1):
                goal[1] += 0.1
                
            goal[0] = FIELD_SIZE
            agent_pos[0] = -2*FIELD_SIZE + (agent_pos[0] + agent_vel[0])
            break
            
        # y座標がグラフ領域を超える
        elif (agent_pos[1] + agent_vel[1]) < -FIELD_SIZE:
            # 超えた時の座標をゴールとする
            goal[1] = -FIELD_SIZE
            agent_pos[1] = 2*FIELD_SIZE + (agent_pos[1] + agent_vel[1])
            break
                                    
        elif (agent_pos[1] + agent_vel[1]) > FIELD_SIZE:
            goal[1] = FIELD_SIZE
            agent_pos[1] = -2*FIELD_SIZE + (agent_pos[1] + agent_vel[1])
            break
            
        # エージェントを初期速度のまま動かす
        agent_pos[0] += agent_vel[0]
        agent_pos[1] += agent_vel[1]

    return goal
    

cdef DTYPE_t[:,:,:] set_goals (
    int num_agents,
    DTYPE_t[:,:,:] all_agents
):
    cdef:
        #DTYPE_t[:,:,:] agent_goals = np.empty([num_agents, 8, 2], dtype=np.float64)
        DTYPE_t[:,:,:] agent_goals = np.zeros([num_agents, 8, 2], dtype=np.float64)
        
        cnp.ndarray[DTYPE_t, ndim=3] all_agents2
        int i, j
        
    # エージェントにゴールを8ずつ設定
    all_agents2 = deepcopy(np.asarray(all_agents))
    for i in range(num_agents):
        for j in range(8):
            goal = calc_goals(all_agents2[i, 1], all_agents2[i, 2])
            agent_goals[i, j, 0] = goal[0]
            agent_goals[i, j, 1] = goal[1]    

    return agent_goals 


cdef DTYPE_t[:,:] calc_distance_all_agents(
    int num_agents,
    DTYPE_t[:,:,:] all_agents
):
    cdef:
        DTYPE_t[:,:] dist_all = np.empty((num_agents, num_agents), dtype=np.float64)
        int i, j
        DTYPE_t dx, dy, dist
        
    for i in range(num_agents):
        for j in range(num_agents):
            dx = all_agents[i, 1, 0] - all_agents[j, 1, 0]
            dy = all_agents[i, 1, 1] - all_agents[j, 1, 1]
            dist = c_norm(dx, dy) - 2*AGENT_SIZE
            dist_all[i, j] = dist

    return dist_all


cdef int[:,:] approach_detect (
    int num_agents,
    DTYPE_t[:,:,:] all_agents,
    DTYPE_t dist
): 
    """ 
    指定した距離より接近したエージェントの数を返す
    ex. self.approach_detect(dist=0.5) -> array([[0, 3],[1, 2],...[24, 1]])
    """
    cdef: 
        int[:,:] approach_agent = np.empty([num_agents, 2], dtype=np.int32)
        DTYPE_t[:,:] dist_all
        int t, i
        int len_visible_agents
        DTYPE_t x
    
    dist_all = calc_distance_all_agents(num_agents, all_agents)
    
    # それぞれのエージェントについて、distより接近したエージェントの数を記録
    for t in range(num_agents):
        visible_agents = [i for i, x in enumerate(dist_all[t]) 
                          if x != -(0.2) and x < dist]
        len_visible_agents = len(visible_agents)
        
        approach_agent[t, 0] = t
        approach_agent[t, 1] = len_visible_agents
                                   
    return approach_agent 


cdef DTYPE_t[:] record_agent_information (
    int num_agents,
    DTYPE_t[:,:,:] all_agents
):
    cdef: 
        DTYPE_t[:] row
        int i
        int[:,:] collision_agent
        
    # 初期の位置と速度を記録
    row = np.concatenate([all_agents[0, 1], all_agents[0, 2]])
    # rowにはある時刻の全エージェントの位置と速度が入る
    for i in range(1, num_agents):
        row = np.concatenate([row, all_agents[i, 1], all_agents[i, 2]])
        
    # エージェントの接近を記録
    # 衝突したエージェント
    collision_agent = approach_detect(num_agents, all_agents, 0)
    for i in range(num_agents):
        row = np.append(row, collision_agent[i, 1])

    # 視野の半分まで接近したエージェント
    collision_agent = approach_detect(num_agents, all_agents, 0.5)
    for i in range(num_agents):
        row = np.append(row, collision_agent[i, 1])
        
    # 視野の4分の1まで接近したエージェント
    collision_agent = approach_detect(num_agents, all_agents, 0.25)
    for i in range(num_agents):
        row = np.append(row, collision_agent[i, 1])

    # 視野の8分の1まで接近したエージェント
    collision_agent = approach_detect(num_agents, all_agents, 0.125)
    for i in range(num_agents):
        row = np.append(row, collision_agent[i, 1])
        
    return row


cdef list find_visible_agents (
    DTYPE_t[:,:] dist_all, 
    int num,
    DTYPE_t[:,:,:] agent_goals,
    DTYPE_t[:,:,:] all_agents,
    int[:] goal_count
):
    """
    エージェント番号numの視野に入った他のエージェントの番号をリストにして返す
    ex. self.find_visible_agents(dist_all, 5) -> [14, 20, 30]
    """
    cdef: 
        list visible_agents
        list near_agents
        int i
        DTYPE_t x
        DTYPE_t angle_difference
        DTYPE_t goal_angle
        DTYPE_t[:] my_pos
        DTYPE_t[:] other_pos
        DTYPE_t[:] goal_pos
        
    # near_agentsは360度の視野に入ったエージェント
    # visible_agentsは視野を狭めた場合に視野に入ったエージェント
    near_agents = [i for i, x in enumerate(dist_all[num]) 
                    if x != -(0.2) and x < VIEW]
    visible_agents = []
                    
    # ゴールベクトルの角度を算出する
    my_pos = agent_goals[num][goal_count[num]]
    goal_pos = all_agents[num][1]
    
    goal_angle = to_deg(calc_rad(my_pos, goal_pos))

    for i in near_agents:
        # 近づいたエージェントとの角度を算出
        other_pos = agent_goals[i][goal_count[i]]
        agent_angle = to_deg(calc_rad(other_pos, goal_pos))
        
        # 近づいたエージェントとの角度とゴールベクトルの角度の差を計算
        angle_difference = abs(goal_angle - agent_angle)
        
        if angle_difference > 180:
            angle_difference = 360 - angle_difference
            
        # 視野に入っているエージェントをvisible_agentsに追加
        if angle_difference <= VIEWING_ANGLE / 2:
            visible_agents.append(i)
    
    return visible_agents


cdef DTYPE_t[:] simple_avoidance (
    int num,
    int num_agents,
    DTYPE_t[:,:,:] all_agents,
    int[:] goal_count,
    DTYPE_t[:,:,:] agent_goals,
    DTYPE_t simple_avoid_vec
):
    cdef: 
        DTYPE_t[:] avoid_vec = np.zeros(2, dtype=np.float64)
        list visible_agents
        int i
        DTYPE_t to_vec1
        DTYPE_t[:] d = np.empty(2, dtype=np.float64)
        DTYPE_t[:,:] dist_all
    
    dist_all = calc_distance_all_agents(num_agents, all_agents)
    visible_agents = find_visible_agents(dist_all, num, agent_goals,
                                         all_agents, goal_count)
    
    if len(visible_agents) == 0:
        return avoid_vec    
    
    ### the followings are simple vectors ###
    for i in visible_agents:
        # dは視界に入ったエージェントに対して反対方向のベクトル        
        d[0] = all_agents[num, 1, 0] - all_agents[i, 1, 0]
        d[1] = all_agents[num, 1, 1] - all_agents[i, 1, 1]
        
        d[0] /= (dist_all[num, i] + 2*AGENT_SIZE)
        d[1] /= (dist_all[num, i] + 2*AGENT_SIZE)
        
        d[0] *= simple_avoid_vec
        d[1] *= simple_avoid_vec

        avoid_vec[0] += d[0] # ベクトルの合成
        avoid_vec[1] += d[1]
        
    # # ベクトルの平均を出す
    # ave_avoid_vec = avoid_vec / len(visible_agents)
    
    # return ave_avoid_vec
    
    # ベクトルを平均しない
    return avoid_vec


cdef DTYPE_t[:] dynamic_avoidance (
    int num,
    int num_agents,
    DTYPE_t[:,:,:] all_agents,
    int[:] goal_count,
    DTYPE_t[:,:,:] agent_goals
):
    """
    動的回避ベクトルの生成
    ex. self.dynamic_avoidance(num=15) -> array([-0.05, 0.2])
    """
    cdef: 
        DTYPE_t[:] avoid_vec = np.zeros(2, dtype=np.float64)
        list visible_agents
        int i
        DTYPE_t to_vec1
        DTYPE_t[:] d = np.empty(2, dtype=np.float64)
        DTYPE_t[:,:] dist_all
        DTYPE_t dist_former
        DTYPE_t t
        DTYPE_t tcpa, dcpa
        DTYPE_t a1, b1, c1, d1
        DTYPE_t braking_index
        cnp.ndarray[DTYPE_t, ndim=1] agent_pos
        cnp.ndarray[DTYPE_t, ndim=1] agent_vel
        cnp.ndarray[DTYPE_t, ndim=1] visible_agent_pos
        cnp.ndarray[DTYPE_t, ndim=1] visible_agent_vel

    dist_all = calc_distance_all_agents(num_agents, all_agents)

    visible_agents = find_visible_agents(dist_all, num, agent_goals,
                                         all_agents, goal_count)
    if len(visible_agents) == 0:
        return avoid_vec    
               
    ### the followings are dynamic vectors ###
    for i in visible_agents:
        # 視野の中心にいるエージェントの位置と速度
        agent_pos = deepcopy(np.asarray(all_agents[num][1]))
        agent_vel = deepcopy(np.asarray(all_agents[num][2]))
        # 視野に入ったエージェントの位置と速度
        visible_agent_pos = deepcopy(np.asarray(all_agents[i, 1]))
        visible_agent_vel = deepcopy(np.asarray(all_agents[i, 2]))
  
        dist_former = dist_all[num][i]
        
        t = 0
        # 2体のエージェントを1ステップ動かして距離を測定
        agent_pos[0] += agent_vel[0]
        agent_pos[1] += agent_vel[1]
        
        visible_agent_pos[0] += visible_agent_vel[0]
        visible_agent_pos[1] += visible_agent_vel[1]
        
        d[0] = agent_pos[0] - visible_agent_pos[0]
        d[1] = agent_pos[1] - visible_agent_pos[1]
        
        dist_latter = c_norm(d[0], d[1]) - 2*AGENT_SIZE
        
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
                t += INTERVAL
                agent_pos[0] += agent_vel[0]
                agent_pos[1] += agent_vel[1]
                visible_agent_pos[0] += visible_agent_vel[0]
                visible_agent_pos[1] += visible_agent_vel[1]
                d[0] = agent_pos[0] - visible_agent_pos[0]
                d[1] = agent_pos[1] - visible_agent_pos[1]
                dist_latter = c_norm(d[0], d[1]) - 2*AGENT_SIZE
                
            if dist_former < 0:
                dcpa = 0 # 最も近い距離で接触している場合はdcpaは0とみなす
            else:
                dcpa = dist_former * 50 # 単位をピクセルに変換
               
            tcpa = t

        # ブレーキ指標の重み
        a1 = -5.145 
        b1 =3.348 
        c1 = 4.286 
        d1 = -13.689   
        
        # ブレーキ指標の算出
        braking_index = (1 / (1 + c_exp(-c1 - d1 * (tcpa/4000)))) * \
                        (1 / (1 + c_exp(-b1 - a1 * (dcpa/50))))
                   
        # dは視界に入ったエージェントに対して反対方向のベクトル
        d[0] = all_agents[num][1][0] - all_agents[i, 1, 0]
        d[1] = all_agents[num][1][1] - all_agents[i, 1, 1]
        
        d[0] /= (dist_all[num][i] + 2 * AGENT_SIZE) # ベクトルの大きさを1にする
        d[1] /= (dist_all[num][i] + 2 * AGENT_SIZE)
        
        d[0] *= braking_index # ブレーキ指標の値を反映
        d[1] *= braking_index
        
        d[0] *= DYNAMIC_AVOID_VEC # ベクトルの最大値を決定
        d[1] *= DYNAMIC_AVOID_VEC
       
        avoid_vec[0] += d[0] # ベクトルの合成
        avoid_vec[1] += d[1]    
        
    # # ベクトルの平均を出す
    # ave_avoid_vec = avoid_vec / len(visible_agents)

    # return ave_avoid_vec
   
    # ベクトルを平均しない
    return avoid_vec


cdef class Simulation:
    cdef:
        int num_steps 
        int num_agents
        int random_seed
        int current_step
        bint returned_results
        
        list data
        int[:] goal_count
        
        DTYPE_t dynamic_percent
        DTYPE_t simple_avoid_vec
        DTYPE_t exe_time
        DTYPE_t[:] completion_time
        DTYPE_t[:] start_step
        DTYPE_t[:] goal_step
        DTYPE_t[:,:] goal_temp
        DTYPE_t[:,:] goal_pos
        DTYPE_t[:,:] start_pos
        DTYPE_t[:,:,:] agent_goals
        DTYPE_t[:,:,:] all_agents
        
    def __cinit__ (self, 
                  int num_steps=500,
                  int num_agents=25, 
                  DTYPE_t dynamic_percent=1.0,
                  DTYPE_t simple_avoid_vec=0.06, 
                  int random_seed=0):
        cdef:
            DTYPE_t[:] row
            
        self.num_steps = num_steps
        self.num_agents = num_agents # エージェント数
        self.dynamic_percent = dynamic_percent # 動的回避を基に回避するエージェントの割合
        self.simple_avoid_vec = simple_avoid_vec # 単純回避での回避ベクトルの大きさ(目盛り)
        self.random_seed = random_seed
                
        self.all_agents = generate_agents(num_agents, dynamic_percent, random_seed)
            
        # エージェントにゴールを8ずつ設定
        self.agent_goals = set_goals(num_agents, self.all_agents)
                
        # ゴールした回数を記録するリスト
        self.goal_count = np.zeros(num_agents, np.int32)
        
        # はみ出た時用のゴール
        self.goal_temp = np.zeros([num_agents, 2])
        
        # 完了時間を測るための変数
        self.start_step = np.zeros([num_agents])
        self.goal_step = np.zeros([num_agents])
        self.goal_pos = np.zeros([num_agents, 2])
        self.start_pos = save_init_pos(self.all_agents, num_agents)
        
        # 完了時間を記録するリスト
        self.completion_time = np.array([])
        
        # 全エージェントの位置と速度、接近を記録
        self.data = []
        row = record_agent_information(self.num_agents, self.all_agents) 
        self.data.append(row) # ある時刻でのエージェントの情報が記録されたrowが集まってdataとなる
        
        self.exe_time = 0
        self.current_step = 0        
        self.returned_results = False
        

    cdef void record_start_and_goal (self, 
        int num
    ):
        """
        完了時間を記録するためのスタート位置とゴール位置を記録
        更新されるパラメータ：
        self.start_pos
        self.goal_pos
        """
        cdef:
            int xy, i_xy, add
        
        # 前回のゴールが左端にあるとき
        if (self.goal_pos[num][0] == -FIELD_SIZE):
            xy, i_xy = 0, 1
            add = 2*FIELD_SIZE
        
        # 前回のゴールが右端にあるとき
        elif (self.goal_pos[num][0] == FIELD_SIZE):
            xy, i_xy = 0, 1
            add = -2*FIELD_SIZE        
        
        # 前回のゴールが下端にあるとき
        elif (self.goal_pos[num][1] == -FIELD_SIZE):
            xy, i_xy = 1, 0
            add = 2*FIELD_SIZE
        
        # 前回のゴールが上端にあるとき
        elif (self.goal_pos[num][1] == FIELD_SIZE):
            xy, i_xy = 1, 0
            add = -2*FIELD_SIZE    
        
        else:
            return 
            
        # スタート位置、ゴール位置を記録
        self.start_pos[num, xy] = self.goal_pos[num, xy] + add
        self.start_pos[num, i_xy] = self.goal_pos[num, i_xy]
        self.goal_pos[num, 0] = self.agent_goals[num][self.goal_count[num]][0] 
        self.goal_pos[num, 1] = self.agent_goals[num][self.goal_count[num]][1]
    
    
    cdef DTYPE_t calc_completion_time (self, 
        int num
    ):
        """ 
        ゴールまで到達した際の完了時間を記録
        ex. self.calc_completion_time(num=10) -> 35.6
        """
        cdef:
            DTYPE_t completion_time
            DTYPE_t to_goal
            DTYPE_t[:] goal_pos
            DTYPE_t dx, dy
        
        # 一回のゴールにおける初めのステップと終わりのステップを記録
        self.start_step[num] = self.goal_step[num] + 1
        self.goal_step[num] = self.current_step
        
        # 一回目のゴール
        if (self.start_step[num] == 1):
            # 一回目のゴールにおける、ゴール位置を記録
            goal_pos = self.agent_goals[num][self.goal_count[num]]
            self.goal_pos[num, 0] = goal_pos[0]
            self.goal_pos[num, 1] = goal_pos[1]
            
        # 一回目以降のゴール
        else:
            self.record_start_and_goal(num)
            
        # 完了時間を計算(ゴールまでのステップ/ゴールまでの距離)
        dx = self.start_pos[num][0] - self.goal_pos[num][0]
        dy = self.start_pos[num][1] - self.goal_pos[num][1]
        to_goal = c_norm(dx, dy)
        #to_goal = np.linalg.norm(np.asarray(self.start_pos[num]) - np.asarray(self.goal_pos[num]))
        completion_time = (self.goal_step[num] - self.start_step[num] + 1) / to_goal
  
        # 外れ値を除外
        if (completion_time > 200):
            return np.nan
    
        return completion_time
    
    
    cdef DTYPE_t calc_remained_completion_time (self, 
        int num
    ):
        """
        1試行が終わり、ゴールに向かう途中の最後の座標から完了時間を算出(やらなくても良いかもしれない)
        """
        cdef:
            DTYPE_t completion_time 
            DTYPE_t a, b, c, d
            DTYPE_t cross_x, cross_y
            DTYPE_t[:] cross
            DTYPE_t distance
        
        # 初めのステップと終わりのステップを記録
        self.start_step[num] = self.goal_step[num] + 1
        self.goal_step[num] = self.num_steps
        
        # ゴールする前に境界をはみ出ている場合
        if not (self.goal_temp[num, 0] == 0 and self.goal_temp[num, 1] == 0):
            # 左右の境界をはみ出た
            if (abs(self.all_agents[num, 1, 0]) > abs(self.all_agents[num, 1, 1])):
                # はみ出る前に戻してあげる
                self.all_agents[num, 1, 0] = -self.all_agents[num, 1, 0]
            # 上下の境界をはみ出た
            elif (abs(self.all_agents[num, 1, 0]) < abs(self.all_agents[num, 1, 1])):
                # はみ出る前に戻してあげる
                self.all_agents[num, 1, 1] = -self.all_agents[num, 1, 1]
            
        self.record_start_and_goal(num)
        
        # スタートからゴールまでの直線の式を計算
        # 傾き
        a = (self.start_pos[num, 1] - self.goal_pos[num, 1]) / \
            (self.start_pos[num, 0] - self.goal_pos[num, 0])
        if a == 0:
            return np.nan
        
        # 切片
        b = -(a * self.start_pos[num, 0]) + self.start_pos[num, 1]
        
        # エージェントの位置を通り、スタートからゴールまでの直線に垂直な直線の式を計算
        # 傾き
        c = (-1) / a
        # 切片
        d = -(c * self.all_agents[num, 1, 0]) + self.all_agents[num, 1, 1]

        # 2つの直線の交点を算出
        cross_x = (b - d) / (-(a - c))
        cross_y = a * cross_x + b
        cross = np.array([cross_x, cross_y])
        
        # スタートから交点までの距離を計算
        distance = np.linalg.norm(self.start_pos[num] - np.asarray(cross))
        if distance == 0:
            return np.nan
        
        # 完了時間を計算(ゴールまでのステップ/ゴールまでの距離)
        completion_time = (self.goal_step[num] - self.start_step[num] + 1) / distance
        
        if (completion_time > 200 or completion_time < 10):
             return np.nan
        
        return completion_time
    
    
    cdef void check_if_goaled (self):    
        cdef:
            int i, xy, i_xy, add
            DTYPE_t pos_x, pos_y, my_posx, my_posy, goal_posx, goal_posy
            DTYPE_t completion_time
            DTYPE_t[:] current_pos
        
        for i in range(self.num_agents):
            pos_x = self.all_agents[i, 1, 0] + self.all_agents[i, 2, 0]
            pos_y = self.all_agents[i, 1, 1] + self.all_agents[i, 2, 1]
            
            # x座標が左端をこえる
            if pos_x < -FIELD_SIZE:
                xy, i_xy = 0, 1
                add = 2*FIELD_SIZE  
            
            # x座標が右端をこえる
            elif pos_x > FIELD_SIZE:
                xy, i_xy = 0, 1
                add = -2*FIELD_SIZE
            
            # y座標が下をこえる
            elif pos_y < -FIELD_SIZE:
                xy, i_xy = 1, 0
                add = 2*FIELD_SIZE
            
            # y座標が上をこえる
            elif pos_y > FIELD_SIZE:
                xy, i_xy = 1, 0
                add = -2*FIELD_SIZE
            
            else:
                continue
                    
            my_posx = self.all_agents[i, 1, 0]
            my_posy = self.all_agents[i, 1, 1]
            goal_posx = self.agent_goals[i][self.goal_count[i]][0]
            goal_posy = self.agent_goals[i][self.goal_count[i]][1]
            
            if (my_posx > goal_posx - 0.1 and 
                my_posx < goal_posx + 0.1 and  
                my_posy > goal_posy - 0.1 and 
                my_posy < goal_posy + 0.1):
                
                # 通常のゴールに到着
                if (self.goal_temp[i, 0] == 0 and self.goal_temp[i, 1] == 0):
                    # 完了時間を算出
                    completion_time = self.calc_completion_time(i)
                    if completion_time:
                        self.completion_time = np.append(
                            self.completion_time, completion_time
                        )
                    self.goal_count[i] += 1 # ゴールした回数を更新
                    
                # はみ出た時用のゴールに到着
                else:
                    self.goal_temp[i, 0] = 0. # はみ出た時用のゴールを初期化
                    self.goal_temp[i, 1] = 0.
                    
            # ゴールに到着せず、境界を超える  
            else:
                # はみ出た時用のゴールが設定されていない
                if (self.goal_temp[i, 0] == 0 and self.goal_temp[i, 1] == 0):
                    # はみ出た時用のゴールを設定
                    self.goal_temp[i][xy] = self.agent_goals[i][self.goal_count[i]][xy] + add
                    self.goal_temp[i][i_xy] = self.agent_goals[i][self.goal_count[i]][i_xy] 
                      
                # はみ出た時用のゴールが設定されている
                else:
                    self.goal_temp[i, 0] = 0. # はみ出た時用のゴールを初期化
                    self.goal_temp[i, 1] = 0.
                    
            current_pos = np.zeros(2)
            current_pos[xy] = self.all_agents[i, 1][xy] + self.all_agents[i, 2][xy] 
            self.all_agents[i, 1][xy] = current_pos[xy] + add
        

    cdef void move_agents (self):
        cdef:
            int i
            DTYPE_t[:] goal_arr
            DTYPE_t[:] row
            DTYPE_t[:] simple_vec
            DTYPE_t[:] dynamic_vec
        
        goal_arr = np.array([GOAL_VEC, 0])
        for i in range(self.num_agents):
            # はみ出た時用のゴールが設定されていない
            # 通常のゴールに向かうベクトルと、回避ベクトルを足したものが速度になる
            if (self.goal_temp[i, 0] == 0 and self.goal_temp[i, 1] == 0):
                rad = calc_rad(self.agent_goals[i][self.goal_count[i]],
                                self.all_agents[i, 1])
                self.all_agents[i, 2] = rotate_vec(goal_arr, rad)     

            # はみ出た時用のゴールが設定されている
            # はみ出た時用のゴールに向かうベクトルと、回避ベクトルを足したものが速度になる
            else:
                rad = calc_rad(self.goal_temp[i], self.all_agents[i, 1])
                self.all_agents[i, 2] = rotate_vec(goal_arr, rad) 
                
            # 回避ベクトルを足す
            if self.all_agents[i, 0, 0] == 0: # 単純回避ベクトルを足す
                simple_vec = simple_avoidance(i, self.num_agents, self.all_agents, 
                                              self.goal_count, self.agent_goals, 
                                              self.simple_avoid_vec)
                self.all_agents[i, 2, 0] += simple_vec[0]
                self.all_agents[i, 2, 1] += simple_vec[1]
                
            elif self.all_agents[i, 0, 0] == 1: # 動的回避ベクトルを足す
                dynamic_vec = dynamic_avoidance(i, self.num_agents, self.all_agents,
                                                self.goal_count, self.agent_goals)
                self.all_agents[i, 2, 0] += dynamic_vec[0]
                self.all_agents[i, 2, 1] += dynamic_vec[1]
                
        self.current_step += 1
        self.check_if_goaled()
        
        for i in range(self.num_agents):
            # 移動後の座標を確定      
            self.all_agents[i, 1, 0] += self.all_agents[i, 2, 0]
            self.all_agents[i, 1, 1] += self.all_agents[i, 2, 1]
            
        row = record_agent_information(self.num_agents, self.all_agents)
        self.data.append(row)
    
    
    cpdef void simulate (self):
        cdef:
            int t
            DTYPE_t start, fin
        
        for t in range(self.num_steps):
            self.move_agents()


    cdef DTYPE_t[:] record_approaches (self,
        str approach_dist
    ):
        """
        エージェント同士が接近した回数を、接近度合い別で出力する
        ex. self.record_approaches('collision', STEP=500, data) -> [0,0,3,...,12]
        """
        cdef: 
            DTYPE_t[:] approach 
            int start
            int stop
            int i
            int total
            
        if approach_dist == 'collision':
            start, stop = 4*self.num_agents, 5*self.num_agents
            
        elif approach_dist == 'half':
            start, stop = 5*self.num_agents, 6*self.num_agents
            
        elif approach_dist == 'quarter':
            start, stop = 6*self.num_agents, 7*self.num_agents
            
        elif approach_dist == 'one_eigth':
            start, stop = 7*self.num_agents, 8*self.num_agents
            
        approach = np.array([])
        for i in range(start, stop, 1):
            total = 0
            for j in range(self.num_steps):
                # 一試行で何回エージェントに衝突したか
                total += self.data[j+1][i]
            
            # 全エージェントの衝突した回数を記録
            approach = np.append(approach, total)
        
        return approach


    def return_results_as_df(self):
        """
        1試行の記録をデータフレームにして返す
        """
        import pandas as pd
        # この関数は破壊的操作で1回目と2回目の値が変化するため、2回目以降の呼び出しを禁止する
        assert self.returned_results == False, "Results have been already returend."
        
        # 最後の座標から完了時間を算出
        for i in range(self.num_agents):
            last_completion_time = self.calc_remained_completion_time(i)
            if last_completion_time:
                self.completion_time = np.append(
                    self.completion_time, last_completion_time
                )
       
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
        completion_time_mean = np.nanmean(self.completion_time)

        # 結果のデータを保存
        dict_result = {
            'time': completion_time_mean,
            'half': half_mean,
            'quarter': quarter_mean,
            'one_eigth': one_eighth_mean,
            'collision': collision_mean,
            'agent': self.num_agents,
            'viewing_angle': VIEWING_ANGLE,
            'num_steps': self.num_steps,
            'dynamic_percent': self.dynamic_percent,
            'simple_avoid_vec': self.simple_avoid_vec,
            'sum_goal_count': np.sum(self.goal_count),
            'exe_time_second': np.round(self.exe_time, 3),
            'exe_time_min': np.round(self.exe_time/60, 3)
        }
        
        df_result = pd.DataFrame(dict_result, index=[f'seed_{self.random_seed}'])
        self.returned_results = True
        
        return df_result
