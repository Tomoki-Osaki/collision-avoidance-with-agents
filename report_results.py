import pandas as pd
import matplotlib.pyplot as plt

# %% increasing agents
df_inc = pd.read_csv('simulation_for_report/increasing_agents.csv')

plt.rcParams['font.size'] = 16
for agent, goal in zip(df_inc.agent, df_inc.sum_goal_count):
    if agent % 2 == 0:
        goal = goal / agent
        plt.scatter(agent, goal, color='tab:blue')
        
#plt.scatter(df_inc.agent, df_inc.sum_goal_count)
#plt.xlabel('Number of agents')
#plt.ylabel("Sum of all agents' goal counts")
#plt.ylabel("Mean of each agent's goal counts")
plt.xlim(0, 155)
#plt.ylim(0, 400)
plt.ylim(0, 4.5)
plt.xticks(range(0, 155, 10))
plt.grid()

# %% merge files of diff seeds of avoid01to09
df_tmp25 = pd.DataFrame()
for i in range(0, 5):
    tmp25 = pd.read_csv(f'simulation_for_report/simple25_avoid01to09_seed{i}.csv')
    df_tmp25 = pd.concat([df_tmp25, tmp25])
df_tmp25 = df_tmp25.drop(['Unnamed: 0', 'awareness'], axis=1)
df_simple25_avoid_01 = df_tmp25.groupby('simple_avoid_vec', as_index=False).mean()

df_tmp50 = pd.DataFrame()
for i in range(0, 3):
    tmp50 = pd.read_csv(f'simulation_for_report/simple50_avoid01to09_seed{i}.csv')
    df_tmp50 = pd.concat([df_tmp50, tmp50])
df_tmp50 = df_tmp50.drop(['Unnamed: 0', 'awareness'], axis=1)
df_simple50_avoid_01 = df_tmp50.groupby('simple_avoid_vec', as_index=False).mean()

# %% comp avoid 0px
df_simple = pd.read_csv('motiduki_results.csv')
df_no_ave25 = pd.read_csv('simulation_for_report/agent25_avoid0.csv')
df_no_ave50 = pd.read_csv('simulation_for_report/agent50_avoid0.csv')

df_awm25 = pd.read_csv('simulation_results/agt25_dynamic_awareness.csv')
df_awm50 = pd.read_csv('simulation_results/agt50_dynamic_awareness.csv')

df_awm25_no_ave = pd.read_csv('simulation_for_report/agent25_awm_not_ave.csv')
df_awm50_no_ave = pd.read_csv('simulation_for_report/agent50_awm_not_ave.csv')

plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'MS Gothic'
plt.scatter(df_simple25_avoid_01['time'], df_simple25_avoid_01['collision'],
            label='25体 単純回避ベクトル: 0.- 0.9 px', color='indigo')
plt.scatter(df_simple.time_25, df_simple.collision_25, 
            label='25体 単純回避ベクトル: 1.0 - 5.0 px', alpha=0.7)
plt.scatter(df_no_ave25['time'].mean(), df_no_ave25['collision'].mean(), 
            #label='25 avoid_vec 0px', 
            color='indigo')
plt.scatter(df_awm25['time'].mean(), df_awm25['collision'].mean(), 
            label='25体 awareness (前回の報告)', color='gray')
plt.scatter(df_awm25_no_ave['time'].mean(), df_awm25_no_ave['collision'].mean(),
            label='25体 awareness (回避ベクトル平均なし)', color='black')

plt.scatter(df_simple50_avoid_01['time'], df_simple50_avoid_01['collision'],
            label='50体 単純回避ベクトル: 0.- 0.9 px', color='orangered', marker='^')
plt.scatter(df_simple.time_50, df_simple.collision_50, 
            label='50体 単純回避ベクトル: 1.0 - 5.0 px', alpha=0.7, marker='^')
plt.scatter(df_no_ave50['time'].mean(), df_no_ave50['collision'].mean(), 
            #label='50 avoid_vec 0px', 
            color='red', marker='^')
plt.scatter(df_awm50['time'].mean(), df_awm50['collision'].mean(), 
            label='50体 awareness (前回の報告)', marker='^', color='gray')
plt.scatter(df_awm50_no_ave['time'].mean(), df_awm50_no_ave['collision'].mean(),
            label='50体 awareness (回避ベクトル平均なし)', marker='^', color='black')

#plt.xlabel('完了時間')
#plt.ylabel('衝\n突\n回\n数', rotation=0)
plt.legend()
plt.grid()

