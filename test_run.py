import yaml
from swarm_environment.env.swarm_environment import SwarmDecisionEnvironment
import numpy as np

# Load configuration
with open("config/configuration.yaml", "r") as f:
    config = yaml.safe_load(f)

# Define lambdas for locations. 
# Lambda = 0.1 means an event happens on average every 10 timesteps.
# Lambda = 0.5 means an event happens on average every 2 timesteps.
num_locations = config["experiment"]["num_locations"]
lambdas = [0.1] * num_locations
if num_locations >= 2:
    lambdas[0] = 0.1  # Sehr sicher
    lambdas[1] = 0.8  # Sehr gefährlich

print("--- Initializing Swarm Environment ---")
env = SwarmDecisionEnvironment(config, lambdas)
env.reset()

print("Environment loaded successfully!")
print(f"Agents: {config['experiment']['num_agents']}")
print(f"Locations: {num_locations}")
print("-" * 50)

# Run 20 random actions
for i in range(20):
    agent_id = env.agent_selection
    
    # Create a random action format: [Destination, Duration]
    # Destination index: 0 to num_locations-1 are Caves, 'num_locations' is the Nest
    random_destination = np.random.randint(0, num_locations + 1)
    random_duration = np.random.randint(10, 50)
    
    # Optional: Mock a random vote change occasionally
    agent_obj = env.get_agent_by_id(agent_id)
    if np.random.random() < 0.3:  # 30% Chance Vote zu ändern
        agent_obj.current_vote = np.random.randint(0, num_locations)

    action = [random_destination, random_duration]
    
    print(f"\n[World Time: {env.current_step:.2f}] Action Prompt:")
    print(f"-> Agent {agent_id} wurde gefragt und wählt: {['Cave 0', 'Cave 1', 'Nest'][random_destination]} für {random_duration} Steps")
    
    # Step ausführen (retourniert Observation für den NÄCHSTEN Agenten)
    obs = env.step(action)
    print(f"Neue Observation erhalten: {obs}")

print("\n🎉 Test Run was completely successful! No crashes!")
