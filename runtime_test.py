import yaml
from swarm_environment.env.swarm_environment import SwarmDecisionEnvironment
import numpy as np
from pettingzoo.test import api_test
import time


with open("config/configuration.yaml", "r") as f:
    config = yaml.safe_load(f)

agents_amount_testing = config["testing"]["agents"]
episodes_amount_testing = config["testing"]["episodes"]
locations_amount_testing = config["testing"]["locations"]

# Override quorum_threshold to 2.0 (200%) so that the swarm never reaches consensus.
# This prevents the environment from terminating early and allows testing the runtime over the full number of steps.
config["experiment"]["quorum_threshold"] = 2.0

with open("tests/runtime_test.csv", "w") as f:
    f.write("agents, episodes, locations, runtime\n")
    for agents_amount in agents_amount_testing:
        config["experiment"]["num_agents"] = agents_amount            
        for episodes_amount in episodes_amount_testing:

            for locations_amount in locations_amount_testing:
                config["experiment"]["num_locations"] = locations_amount
                
                num_locations = config["experiment"]["num_locations"]

                #print("--- Initializing Swarm Environment ---")

                start_time = time.perf_counter()
                env = SwarmDecisionEnvironment(config)
                env.reset()

                for i in range(episodes_amount):
                    agent_id = env.agent_selection
                    if agent_id is None or len(env.agents) == 0:
                        # Episode naturally finished because all agents terminated/truncated
                        break
                    if(not agent_id in env.agents):
                        print(f"Round crashed for: Agents: {agents_amount}, Episodes: {episodes_amount}, Locations: {locations_amount}")
                        print(f"  - agent_id: {agent_id}")
                        print(f"  - env.agents: {env.agents}")
                        print(f"  - env.terminations: {env.terminations}")
                        print(f"  - env.truncations: {env.truncations}")
                        print(f"  - env.prio_Q size: {len(env.prio_Q.Q)}")
                        break
                    obs, reward, termination, truncation, info = env.last()
                    
                    #print(f"\n[World Time: {env.current_step:.2f}] Agent {agent_id} ist an der reihe")
                    
                    if termination or truncation:
                        action = None
                        #print(f"Agent {agent_id} hat terminiert")
                    else:
                        random_destination = np.random.randint(0, num_locations + 1)
                        random_duration = np.random.randint(10, 50)
                        
                        agent_obj = env.get_agent_by_id(agent_id)
                        if np.random.random() < 0.3:
                            agent_obj.current_vote = np.random.randint(0, num_locations)

                        action = [random_destination, random_duration]
                        
                        #print(f"Action Prompt: Agent wählt {random_destination} für {random_duration} Steps")

                    env.step(action)
                end_time = time.perf_counter()
                runtime_in_ms = (end_time - start_time) * 1000

                print(f"{agents_amount}, {episodes_amount}, {locations_amount}, {runtime_in_ms}\n")
                f.write(f"{agents_amount}, {episodes_amount}, {locations_amount}, {runtime_in_ms}\n")
            
print("Test run complete!")
