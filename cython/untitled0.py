import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
import time 
import sim_without_awm as sw
import classSimulation as cs
import funcSimulation as fs
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

def wd():
    import os
    try: os.chdir('C:/Users/ootmo/Downloads/cython')
    except FileNotFoundError: pass
wd()

def plot_pos(sim_obj):
    fig, ax = plt.subplots(figsize=(8,8))
    for i in range(sim_obj.num_agents):
        x = sim_obj.all_agents[i][1][0]*50+250
        y = sim_obj.all_agents[i][1][1]*50+250
        ax.add_artist(Circle((x, y), radius=5, color='blue', alpha=0.6))
        #ax.scatter(x, y, color='blue')
        ax.annotate(i, xy=(x, y))
        
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)
    ax.set_xticks(range(0, 501, 50))
    ax.set_yticks(range(0, 501, 50))
    ax.grid()
    plt.show()
    
# %%
import try_cpp_libs as tcl
arr = tcl.generate_random_numbers(1)
print(np.asarray(arr))    

rng_mt = np.random.Generator(np.random.MT19937(1))
arr2 = rng_mt.uniform(-5, 5, 1)
arr3 = rng_mt.uniform(-5, 5, 1)

arr2d = tcl.create_2d_array(rows=2, cols=2, initial_value=0)

# %%
import pandas as pd
res150 = pd.read_csv("D:/simulation_for_report/increasing_agents.csv")


num_agents = 25
num_steps = 500
dyn_per = 1.0

import all_func as af

sim = af.Simulation(num_agents=num_agents, num_steps=num_steps, dynamic_percent=1.0)

# print(np.asarray(sim.all_agents[0]))
# dist = sim.calc_distance_all_agents()
# print(np.asarray(dist))
# print(np.asarray(sim.find_visible_agents(dist, 3)))
# vis_agents = np.asarray(sim.find_visible_agents(dist, 3))
# sim.record_start_and_goal(3)
# np.asarray(sim.start_pos)
# np.asarray(sim.goal_pos)

# print(np.asarray(sim.approach_detect(0.5)))
# print(np.asarray(sim.record_agent_information()))
# print(np.asarray(sim.calc_completion_time(3)))
# print(np.asarray(sim.calc_remained_completion_time(3)))
# sim.check_if_goaled()
# print(np.asarray(sim.all_agents[1]))
# print(np.asarray(sim.goal_temp))
# print(np.asarray(sim.simple_avoidance(0)))

print(np.asarray(sim.dynamic_avoidance(0)))

sim.move_agents()
print(np.asarray(sim.all_agents[0]))

start = time.perf_counter()
sim = af.Simulation(num_agents=num_agents, num_steps=num_steps, dynamic_percent=0.5)
sim.simulate()
exe = time.perf_counter() - start
print('cy exe', exe)
print(np.asarray(sim.all_agents[0]))
df = sim.return_results_as_df()

import untitled2 as u2
u2t = u2.Simulation(num_agents=num_agents, num_steps=num_steps, dynamic_percent=dyn_per)

start = time.perf_counter()
u2t.simulate()
exe = time.perf_counter() - start
print('\nexe', exe)
df = u2t.return_results_as_df()
print(df)

import cythonized_simulation as cys
cyst = cys.Simulation(num_agents=num_agents, num_steps=num_steps, dynamic_percent=dyn_per)
cyst.simulate()
print(cyst.return_results_as_df())

# %%
t = cs.Simulation(num_agents=num_agents, num_steps=num_steps, dynamic_percent=1.0)
# print('p', t.all_agents[0]['p'])
# print('v', t.all_agents[0]['v'])
# dist_t = t.calc_distance_all_agents()

# t.find_visible_agents(dist_t, 3)
# #fs.plot_positions(t)

# t.record_start_and_goal(3)
# t.start_pos
# t.goal_pos
# print(t.approach_detect(0.5))
# print(t.record_agent_information())
# print(t.calc_completion_time(3))
# print(t.calc_remained_completion_time(3, num_steps))
# t.check_if_goaled()
# print(t.all_agents[1]['p'])
# print(t.all_agents[1]['v'])
# print(t.goal_temp)
# print(t.simple_avoidance(0))

print(t.dynamic_avoidance(0))

t.move_agents()
print(t.all_agents[0]['p'])
print(t.all_agents[0]['v'])

t = cs.Simulation(num_agents=num_agents, num_steps=num_steps, dynamic_percent=dyn_per)
start = time.perf_counter()
t.simulate()
exe = time.perf_counter() - start
print('\nexe', exe)
print(t.all_agents[0]['p'])
print(t.all_agents[0]['v'])
df_t = t.return_results_as_df()
print(df_t)


swt = sw.Simulation(num_agents=num_agents, num_steps=num_steps, dynamic_percent=dyn_per)
start = time.perf_counter()
swt.simulate()
exe = time.perf_counter() - start
print('\nexe', exe)
print(swt.all_agents[0]['p'])
print(swt.all_agents[0]['v'])
df_swt = swt.return_results_as_df()
print(df_swt)

# %%
import retry 
ret = retry.Simulation(num_agents=5, dynamic_percent=0.)
plot_pos(ret)

start = time.perf_counter()
for _ in range(500):
    ret.move_agents()
exe = time.perf_counter() - start
print('exe', exe)



t = cs.Simulation(num_agents=50, num_steps=100, dynamic_percent=0.)
print('p', t.all_agents[0]['p'])
print('v', t.all_agents[0]['v'])
dist_t = t.calc_distance_all_agents()

t.find_visible_agents(dist_t, 20)

fs.plot_positions(t)
start = time.perf_counter()
t.simulate()
exe = time.perf_counter() - start
print('exe', exe)

print(t.return_results_as_df())

swt = sw.Simulation(num_agents=25, num_steps=100)
print('p', swt.all_agents[0]['p'])
print('v', swt.all_agents[0]['v'])
#fs.plot_positions(swt, swt.current_step)

swt.simulate()
res = swt.return_results_as_df()
print(res)
#res.to_csv('for_ref.csv')
    

# %%
num_agents = 5
fig, ax = plt.subplots(figsize=(8,8))
for i in range(num_agents):
    x = sim.all_agents[i][1][0]*50+250
    y = sim.all_agents[i][1][1]*50+250
    ax.add_artist(Circle((x, y), radius=5, color='blue', alpha=0.6))
    #ax.scatter(x, y, color='blue')
    ax.annotate(i, xy=(x, y))
    
ax.set_xlim(0, 500)
ax.set_ylim(0, 500)
ax.set_xticks(range(0, 501, 50))
ax.set_yticks(range(0, 501, 50))
ax.grid()
plt.show()

# %%
arr1 = np.array([1, 2, 3])

def arr(arr2):
    arr2[0] = 100
    
    return arr2
   
arr(arr1) 


