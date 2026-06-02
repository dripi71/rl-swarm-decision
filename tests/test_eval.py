import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml



with open("config/configuration.yaml", "r") as f:
    config = yaml.safe_load(f)

agents_amount_testing = config["testing"]["agents"]
episodes_amount_testing = config["testing"]["episodes"]
locations_amount_testing = config["testing"]["locations"]

df = pd.read_csv("tests/runtime_test.csv", sep=", ")

df_filtered_ep = df[df["locations"] == 2]
df_filtered_ep = df_filtered_ep[df_filtered_ep["agents"].isin(agents_amount_testing)]
df_filtered_ep["agents"] = df_filtered_ep["agents"].astype(str)
legend_order = [str(x) for x in agents_amount_testing]

plt.figure(figsize=(10, 6))
sns.lineplot(data=df_filtered_ep, x="episodes", y="runtime", hue="agents", hue_order=legend_order, marker="o", palette="viridis")

plt.title("Runtime in dependency of agents and episodes (location fixed = 2)")
plt.xlabel("Number of Episodes")
plt.ylabel("Runtime (ms)")
plt.legend(title="Number of Agents")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()


df_filtered_agents = df[df["agents"] == 200]
df_filtered_agents = df_filtered_agents[df_filtered_agents["locations"].isin(locations_amount_testing)]
df_filtered_agents["locations"] = df_filtered_agents["locations"].astype(str)
legend_order = [str(x) for x in locations_amount_testing]

plt.figure(figsize=(10,6))
sns.lineplot(data=df_filtered_agents, x="episodes", y="runtime", hue="locations", hue_order=legend_order, marker="o", palette="viridis")

plt.title("Runtime in dependency of locations and episodes (agents fixed = 200)")
plt.xlabel("Number of Episodes")
plt.ylabel("Runtime (ms)")
plt.legend(title="Number of Locations")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()


df_filtered_episodes = df[df["episodes"] == 100]
df_filtered_episodes = df_filtered_episodes[df_filtered_episodes["locations"].isin(locations_amount_testing)]
df_filtered_episodes["locations"] = df_filtered_episodes["locations"].astype(str)
legend_order = [str(x) for x in locations_amount_testing]

plt.figure(figsize=(10,6))
sns.lineplot(data=df_filtered_episodes, x="agents", y="runtime", hue="locations", hue_order=legend_order, marker="o", palette="viridis")

plt.title("Runtime in dependency of agents and locations (episodes fixed = 100)")
plt.xlabel("Number of Agents")
plt.ylabel("Runtime (ms)")
plt.legend(title="Number of Locations")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()