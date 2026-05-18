import yaml
from swarm_environment.env.swarm_environment import SwarmDecisionEnvironment
import numpy as np
from pettingzoo.test import api_test

with open("config/configuration.yaml", "r") as f:
    config = yaml.safe_load(f)
num_locations = config["experiment"]["num_locations"]
lambdas = [0.1] * num_locations
if num_locations >= 2:
    lambdas[0] = 0.1  
    lambdas[1] = 0.8  

print("--- Initializing Swarm Environment ---")
env = SwarmDecisionEnvironment(config, lambdas)
api_test(env, num_cycles=1000)


env.reset()

for i in range(100):
    agent_id = env.agent_selection
    
    obs, reward, termination, truncation, info = env.last()
    
    print(f"\n[World Time: {env.current_step:.2f}] Agent {agent_id} ist an der reihe")
    print(f"Aktuelle Observation erhalten: {obs}")
    
    if termination or truncation:
        action = None
        print(f"Agent {agent_id} hat terminiert")
    else:
        random_destination = np.random.randint(0, num_locations + 1)
        random_duration = np.random.randint(10, 50)
        
        agent_obj = env.get_agent_by_id(agent_id)
        if np.random.random() < 0.3:
            agent_obj.current_vote = np.random.randint(0, num_locations)

        action = [random_destination, random_duration]
        
        print(f"Action Prompt: Agent wählt {random_destination} für {random_duration} Steps")
    
    print(f"Prio Q:")
    print(env.prio_Q.print())

    env.step(action)

print("Test run complete!")
