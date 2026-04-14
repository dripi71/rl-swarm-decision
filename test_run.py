import yaml
from swarm_environment.env.swarm_environment import SwarmDecisionEnvironment
import numpy as np

with open("config/configuration.yaml", "r") as f:
    config = yaml.safe_load(f)
num_locations = config["experiment"]["num_locations"]
lambdas = [0.1] * num_locations
if num_locations >= 2:
    lambdas[0] = 0.1  
    lambdas[1] = 0.8  

print("--- Initializing Swarm Environment ---")
env = SwarmDecisionEnvironment(config, lambdas)
env.reset()

print("Environment loaded successfully!")
print(f"Agents: {config['experiment']['num_agents']}")
print(f"Locations: {num_locations}")
print("-" * 50)

for i in range(20):
    agent_id = env.agent_selection
    
    random_destination = np.random.randint(0, num_locations + 1)
    random_duration = np.random.randint(10, 50)
    
    agent_obj = env.get_agent_by_id(agent_id)
    if np.random.random() < 0.3:
        agent_obj.current_vote = np.random.randint(0, num_locations)

    action = [random_destination, random_duration]
    
    print(f"\n[World Time: {env.current_step:.2f}] Action Prompt:")
    print(f"-> Agent {agent_id} wurde gefragt und wählt: {['Cave 0', 'Cave 1', 'Nest'][random_destination]} für {random_duration} Steps")
    
    obs = env.step(action)
    print(f"Neue Observation erhalten: {obs}")

print("\n🎉 Test Run was completely successful! No crashes!")
