# 作成者： 望月さん

# %% TO DO
# まずアルゴリズムの仕様を理解する
# 後に、判断エントロピーやawarenessモデルをどのように組み込めば思いやりを実現できるかを考える

# %%
import numpy as np
import pandas as pd
import matplotlib.animation as animation
import tqdm
import simfuncs
from simfuncs import AGENT, COLOR, TRIAL, STEP, INTERVAL

# %%
fig, ax = simfuncs.define_fig_ax()

# 各シミュレーションの結果を保存する変数
row_label = []
values = []

# 図のデータをまとめるもの、これを流すことでアニメーションになる
ims = []

for num in range(TRIAL):
    np.random.seed(num)
    O = simfuncs.simulation(method='simple', viewing_angle=360)

    data = []
    column_label = []
    index_label = []
    index_label.append('initial_value')

    # 出力データの列のラベルを作成
    for i in range(AGENT):
        column_label.append('agent ' + str(i) + ' pos_x')
        column_label.append('agent ' + str(i) + ' pos_y')
        column_label.append('agent ' + str(i) + ' vel_x')
        column_label.append('agent ' + str(i) + ' vel_y')
    
    # 衝突(距離0以下)
    for i in range(AGENT):
        column_label.append('agent ' + str(i) + ' collision')
    # 視野の半分の距離まで接近
    for i in range(AGENT):
        column_label.append('agent ' + str(i) + ' half')
    # 視野の四分の一の距離まで接近
    for i in range(AGENT):
        column_label.append('agent ' + str(i) + ' quarter')
    # 視野の八分の一の距離まで接近
    for i in range(AGENT):
        column_label.append('agent ' + str(i) + ' one_eighth')

    # 初期の位置と速度を記録
    row = []
    row = np.concatenate([O.all_agent[0]['p'], O.all_agent[0]['v']])
    # rowにはある時刻の全エージェントの位置と速度が入る
    for i in range(1, AGENT):
        row = np.concatenate([row, O.all_agent[i]['p'], O.all_agent[i]['v']])


    # 衝突したエージェントの数を記録
    collision_agent = O.approach_detect(0)
    for i in range(AGENT):
        row = np.append(row, collision_agent[i])
    
    # 視野の半分より接近したエージェントの数を記録
    collision_agent = O.approach_detect(0.5)
    for i in range(AGENT):
        row = np.append(row, collision_agent[i])
    
    # 視野の四分の一より接近したエージェントの数を記録
    collision_agent = O.approach_detect(0.25)
    for i in range(AGENT):
        row = np.append(row, collision_agent[i])
        
    # 視野の八分の一より接近したエージェントの数を記録
    collision_agent = O.approach_detect(0.125)
    for i in range(AGENT):
        row = np.append(row, collision_agent[i])
        
    # ある時刻でのエージェントの情報が記録されたrowが集まってdataとなる
    data.append(row)

    # 最初に表示する図の作成
    plot_data = O.showImage()
    im = ax.scatter(*plot_data, marker="o", s=40, c=COLOR)
    ims.append([im])


    # シミュレーション
    for t in tqdm.tqdm(range(STEP)):
        O.simulate(t + 1)
        index_label.append(t + 1)
        # どこまで進んでいるか分かる用、numは試行回数でt+1はステップ数
        #print(str(num) + "  " + str(t+1))

        # シミュレーションごとに値を記録
        row = []
        row = np.concatenate([O.all_agent[0]['p'], O.all_agent[0]['v']])
        for i in range(1, AGENT):
            row = np.concatenate([row, O.all_agent[i]['p'], O.all_agent[i]['v']])

        # 衝突したエージェントを記録
        collision_agent = O.approach_detect(0)
        for i in range(AGENT):
            row = np.append(row, collision_agent[i])

        collision_agent = O.approach_detect(0.5)
        for i in range(AGENT):
            row = np.append(row, collision_agent[i])
            
        collision_agent = O.approach_detect(0.25)
        for i in range(AGENT):
            row = np.append(row, collision_agent[i])

        collision_agent = O.approach_detect(0.125)
        for i in range(AGENT):
            row = np.append(row, collision_agent[i])

        data.append(row)

        # 図を作成
        plot_data = O.showImage()
        im = ax.scatter(*plot_data, marker="o", s=40, c=COLOR)
        ims.append([im])
        
        goal_step = STEP
        
    # csvとしてステップごとに位置、速度、接近した回数を記録
    """
    df = pd.DataFrame(data, columns=column_label, index=index_label)
    df.to_csv('/Users/mango/卒業研究/to_csv_out_' + str(num) + '.csv')
    """
    
    # 最後の座標から完了時間を算出
    for i in range(AGENT):
        last_completion_time = O.calc_last_completion_time(i)
        if not last_completion_time == None:
            O.completion_time.append(last_completion_time)

    # 完了時間をまとめたファイルを作成
    column_label = ["completion_time"]
    index_label = []
    
    for i in range(len(O.completion_time)):
        index_label.append(i + 1)
    
    """
    df = pd.DataFrame(O.completion_time, columns=column_label, index=index_label)
    df.to_csv('/Users/mango/卒業研究/to_csv_out_completion_time_' + str(num) + '.csv')
    """
    
    # 加速度はx, y速度の差からなるベクトルの大きさ
    accel = []
    agents_accels = []
    for i in range(2, 4*AGENT+2, 4):
        agent_accels = []
        for j in range(goal_step - 1):
            # x軸方向の速度の差分
            x_accel = abs((data[j+1][i] - data[j+2][i]) * 50)
            # y軸方向の速度の差分
            y_accel = abs((data[j+1][i+1] - data[j+2][i+1]) * 50)
            # x, y速度の差からなるベクトルの大きさ
            temp = np.sqrt(x_accel ** 2 + y_accel ** 2)
            agent_accels.append(temp)
            
        # 全エージェントの加速度を記録
        agents_accels.append(agent_accels)
        
        # 全エージェントの加速度の総和を記録
        accel.append(np.sum(agent_accels))

        
    # 加速度をまとめたファイル
    column_label = []
    for i in range(AGENT):
        column_label.append('agent ' + str(i) + ' accel')

    index_label = []
    for i in range(len(agents_accels[0])):
        index_label.append(i + 1)
    
    df = pd.DataFrame(agents_accels, columns=index_label, index=column_label)
    df = df.T
    # df.to_csv('/Users/mango/卒業研究/to_csv_out_accel_' + str(num) + '.csv')

   
    # 衝突した数
    collision = []
    for i in range(4*AGENT, 5*AGENT, 1):
        sum = 0
        for j in range(goal_step):
            # 一試行で何回エージェントに衝突したか
            sum += data[j+1][i]
        
        # 全エージェントの衝突した回数を記録
        collision.append(sum)
        
    # 視野の半分に接近したエージェントの数
    half = []
    for i in range(5*AGENT, 6*AGENT, 1):
        sum = 0
        for j in range(goal_step):
            sum += data[j+1][i]
        
        half.append(sum)
        
    # 視野の四分の一に接近した回数
    quarter = []
    for i in range(6*AGENT, 7*AGENT, 1):
        sum = 0
        for j in range(goal_step):
            sum += data[j+1][i]
        
        quarter.append(sum)

    # 視界の八分の一に接近した回数
    one_eighth = []
    for i in range(7*AGENT, 8*AGENT, 1):
        sum = 0
        for j in range(goal_step):
            sum += data[j+1][i]
        
        one_eighth.append(sum)
    
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
                   one_eighth_mean, collision_mean, O.viewing_angle, O.method])

# 値をまとめたcsvファイルの作成
column_label = ['accel', 'time', 'half', 'quarter', 'one_eighth', 
                'collision', 'viewing_angle', 'method']
                      
df = pd.DataFrame(values, columns=column_label, index=row_label)
# 保存する場所は自由に決めてください
df.to_csv(f'{O.method}_{str(O.viewing_angle)}.csv')
print(df) # show results

# %% plot animation
ani = animation.ArtistAnimation(fig, ims, interval=INTERVAL, repeat=False)
ani.save(f'ani_{O.method}.gif')
