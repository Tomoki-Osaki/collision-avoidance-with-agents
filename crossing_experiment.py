import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

df = pd.read_csv("C:/Users/ootmo/OneDrive/Desktop/修論/トヨタ_共同プロジェクト/collision-avoidance-with-agents/glob_shaped/20220405_all_group_ha_keiro1_index_1_8_11_to_0_2.bag.csv")
for col in df.columns: print(col)

plt.plot(df['/vrpn_client_node/body_0/pose/field.pose.position.x'],
         df['/vrpn_client_node/body_0/pose/field.pose.position.z'])
plt.plot(df['/vrpn_client_node/body_1/pose/field.pose.position.x'],
         df['/vrpn_client_node/body_1/pose/field.pose.position.z'])

for x1, y1, x2, y2 in zip(
        df['/vrpn_client_node/body_0/pose/field.pose.position.x'],
        df['/vrpn_client_node/body_0/pose/field.pose.position.z'],
        df['/vrpn_client_node/body_1/pose/field.pose.position.x'],
        df['/vrpn_client_node/body_1/pose/field.pose.position.z']
        ):
    plt.scatter(x1, y1, color='red')
    plt.scatter(x2, y2, color='blue')
plt.show()


tcpa = 0 # Time to Closest of Point of Approach
dcpa = 0 # Distance at Closest Point of Approach

def calc_braking_rate(a1=-5.145, b1=3.348, c1=4.286, d1=-13.689):
    braking_index = (1 / (1 + np.exp(-c1 - d1 * (tcpa/4000)))) * \
                    (1 / (1 + np.exp(-b1 - a1 * (dcpa/50))))
                    
    return braking_index

pos_data = zip(df['/vrpn_client_node/body_0/pose/field.pose.position.x'],
               df['/vrpn_client_node/body_0/pose/field.pose.position.z'],
               df['/vrpn_client_node/body_1/pose/field.pose.position.x'],
               df['/vrpn_client_node/body_1/pose/field.pose.position.z'])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)

def update(frame):
    ax.scatter(frame[0], frame[1], color='red')
    ax.scatter(frame[2], frame[3], color='blue')
    
anim = FuncAnimation(fig, update, frames=pos_data, interval=200)

anim.save("c03.gif", writer='pillow')
plt.close()

