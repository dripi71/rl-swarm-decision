import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

# Graphs: Time on y-axis
#         Agents on x-axis
#         Epsisodes as colors
#         locations represented in other graph
#  Four different plots with fixed locations: [2,10,50,100]

with open("config/configuration.yaml", "r") as f:
    config = yaml.safe_load(f)

agents_amount_testing = config["testing"]["agents"]
episodes_amount_testing = config["testing"]["episodes"]
locations_amount_testing = config["testing"]["locations"]

df = pd.read_csv("tests/runtime_test.csv", sep=",")

for location_amount in locations_amount_testing:
    
    df_filtered_ep = df[df["locations"] == int(location_amount)].copy()

    print(df_filtered_ep)
    df_filtered_ep = df_filtered_ep[df_filtered_ep["episodes"].isin(episodes_amount_testing)]
    df_filtered_ep["episodes"] = df_filtered_ep["episodes"].astype(str)
    legend_order = sorted([int(x) for x in episodes_amount_testing])

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_filtered_ep, x="agents", y="runtime", hue="episodes", hue_order=legend_order, marker="o", palette="viridis")

    plt.title(f"Runtime in dependency of agents and episodes (location fixed = {location_amount})")
    plt.xlabel("Number of Agents")
    plt.ylabel("Runtime (ms)")
    plt.legend(title="Number of Episodes")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(f"tests/Runtime_agents_location_{location_amount}.png")
    plt.close()