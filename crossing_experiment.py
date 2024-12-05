import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/ootmo/OneDrive/Desktop/修論/トヨタ_共同プロジェクト/collision-avoidance-with-agents/glob_shaped/20220405_all_group_ha_keiro1_index_1_8_11_to_0_2.bag.csv")

plt.plot(df['/vrpn_client_node/body_0/pose/field.pose.position.x'],
         df['/vrpn_client_node/body_0/pose/field.pose.position.z'])
plt.plot(df['/vrpn_client_node/body_1/pose/field.pose.position.x'],
         df['/vrpn_client_node/body_1/pose/field.pose.position.z'])

for x, y in zip(df['/vrpn_client_node/body_0/pose/field.pose.position.x'],
                df['/vrpn_client_node/body_0/pose/field.pose.position.z']):
    plt.scatter(x, y)
plt.show()
