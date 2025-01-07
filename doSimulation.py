""" 
2025/01/07 
シミュレーションを実際に実行する
"""
# %%
import numpy as np
import pandas as pd
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import tqdm, gc
import classSimulation as cs
import funcSimulation as fs

# %% global variables

# グラフの目盛りの最大値・最小値
FIELD_SIZE = 5 
# 目盛りは最大値5、最小値-5で10目盛り
# グラフ領域の幅と高さは500pxなので、1pxあたり0.02目盛りとなる

TRIAL = 1 # 試行回数
STEP = 50 # 1回の試行(TRIAL)で動かすステップの回数

# %% シミュレーションの前準備
fig, ax = fs.define_fig_ax(width=500, height=500, field_size=FIELD_SIZE)

# 各シミュレーションの結果を保存する変数
row_label = []
values = []
ims = [] # 図のデータをまとめるもの、これを流すことでアニメーションになる

# %% シミュレーション
# 一度にSTEP数simaulateメソッドを使用するシミュレーションを、TRIALの回数行う
for num in range(TRIAL):
    np.random.seed(num)
    O = cs.Simulation(interval=100, 
                      agent_size=0.1, agent=25, 
                      view=1, viewing_angle=360, 
                      goal_vec=0.06,  
                      avoidance='simple',
                      simple_avoid_vec=0.06, 
                      dynamic_avoid_vec=0.06)

    data = []
    column_label = []
    index_label = ['initial_value']

    # 出力データの列のラベルを作成
    for i in range(O.agent):
        column_label.append('agent ' + str(i) + ' pos_x')
        column_label.append('agent ' + str(i) + ' pos_y')
        column_label.append('agent ' + str(i) + ' vel_x')
        column_label.append('agent ' + str(i) + ' vel_y')
        column_label.append('agent ' + str(i) + ' collision') # 衝突(距離0以下)
        column_label.append('agent ' + str(i) + ' half') # 視野の半分の距離まで接近
        column_label.append('agent ' + str(i) + ' quarter') # 視野の四分の一の距離まで接近
        column_label.append('agent ' + str(i) + ' one_eighth') # 視野の八分の一の距離まで接近


    row = O.record_agent_information() # 全エージェントの位置と速度、接近を記録
    # ある時刻でのエージェントの情報が記録されたrowが集まってdataとなる
    data.append(row)

    # 最初に表示する図の作成
    plot_data = O.show_image()
    im = ax.scatter(*plot_data, s=40, marker="o", c='blue')
    ims.append([im])

    ##### シミュレーション (STEP数だけ繰り返す) #####
    for t in tqdm.tqdm(range(STEP)):
        O.simulate(now_step=t+1)
        index_label.append(t + 1)

        # シミュレーションごとに値を記録
        row = O.record_agent_information()
        data.append(row)

        # 図を作成
        plot_data = O.show_image()
        im = ax.scatter(*plot_data, s=40, marker="o", c='blue')
        ims.append([im])
        
    ##### シミュレーション終了 ######    
        
    
    ##### シミュレーションで得たデータを記録 #####
    # csvとしてステップごとに位置、速度、接近した回数を記録
    """
    df = pd.DataFrame(data, columns=column_label, index=index_label)
    df.to_csv('to_csv_out_' + str(num) + '.csv')
    """
    
    # 最後の座標から完了時間を算出
    for i in range(O.agent):
        last_completion_time = O.calc_last_completion_time(i, STEP)
        if not last_completion_time == None:
            O.completion_time.append(last_completion_time)

    # 完了時間をまとめたファイルを作成
    column_label = ["completion_time"]
    index_label = []
    
    for i in range(len(O.completion_time)):
        index_label.append(i + 1)
    
    """
    df = pd.DataFrame(O.completion_time, columns=column_label, index=index_label)
    df.to_csv('to_csv_out_completion_time_' + str(num) + '.csv')
    """
    
    # 加速度はx, y速度の差からなるベクトルの大きさ
    accel = []
    agents_accels = []
    for i in range(2, 4*O.agent+2, 4):
        agent_accels = []
        for j in range(STEP - 1):
            # x軸方向の速度の差分
            x_accel = abs((data[j+1][i] - data[j+2][i]) * 50)
            # y軸方向の速度の差分
            y_accel = abs((data[j+1][i+1] - data[j+2][i+1]) * 50)
            # x, y速度の差からなるベクトルの大きさ
            temp = np.sqrt(x_accel**2 + y_accel**2)
            agent_accels.append(temp)
            
        # 全エージェントの加速度を記録
        agents_accels.append(agent_accels)
        
        # 全エージェントの加速度の総和を記録
        accel.append(np.sum(agent_accels))

        
    # 加速度をまとめたファイル
    column_label = []
    for i in range(O.agent):
        column_label.append('agent ' + str(i) + ' accel')

    index_label = []
    for i in range(len(agents_accels[0])):
        index_label.append(i + 1)
    
    df = pd.DataFrame(agents_accels, columns=index_label, index=column_label)
    df = df.T
    # df.to_csv('to_csv_out_accel_' + str(num) + '.csv')
   
    # 衝突した数、視野の半分、視野の四分の一、視野の八分の一に接近した回数
    collision = O.record_approaches('collision', STEP, data)
    half =  O.record_approaches('half', STEP, data)
    quarter =  O.record_approaches('quarter', STEP, data)
    one_eighth =  O.record_approaches('one_eigth', STEP, data)
    
    # 各指標の平均を計算
    accel_mean = np.mean(accel)
    collision_mean = np.mean(collision)
    half_mean = np.mean(half)
    quarter_mean = np.mean(quarter)
    one_eighth_mean = np.mean(one_eighth)
    completion_time_mean = np.mean(O.completion_time)

    
    # 各試行の結果のデータを保存
    row_label.append('seed_' + str(num))
    values.append([accel_mean, completion_time_mean, half_mean, quarter_mean, 
                   one_eighth_mean, collision_mean, O.agent, O.viewing_angle, 
                   STEP, O.avoidance])

# 値をまとめたcsvファイルの作成
column_label = ['accel', 'time', 'half', 'quarter', 'one_eighth', 'collision', 
                'agent', 'viewing_angle', 'step', 'method']
                      
df = pd.DataFrame(values, columns=column_label, index=row_label)
# 保存する場所は自由に決めてください
df.to_csv(f'{O.avoidance}_ang{O.viewing_angle}_agt{O.agent}_stp{STEP}.csv')
print(df) # show results

# ani = animation.ArtistAnimation(fig, ims, interval=O.interval, repeat=False)
# ani.save(f'ani_{O.avoidance}_ang{O.viewing_angle}_agt{O.agent}_stp{STEP}.mp4')
