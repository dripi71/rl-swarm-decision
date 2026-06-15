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
                num_locations = locations_amount

                #print("--- Initializing Swarm Environment ---")

                start_time = time.perf_counter()           
                env = SwarmDecisionEnvironment(config)

                for i in range(episodes_amount):
                    lambdas = np.random.uniform(0.01, 1.0, num_locations)
                    env.reset(lambdas=lambdas) 

                    while True:
                        agent_id = env.agent_selection
                        if agent_id is None or len(env.agents) == 0:
                            # Episode naturally finished because all agents terminated/truncated
                            break
                        obs, reward, termination, truncation, info = env.last()
                                                
                        if termination or truncation:
                            action = None
                        else:
                            random_destination = np.random.randint(0, num_locations + 1)
                            random_dur_params = np.random.uniform(-3.0, 3.0, size=2).astype(np.float32)
                            random_vote = np.random.randint(0, num_locations + 1)

                            action = {
                                "location":        np.int64(random_destination),
                                "duration_params":  random_dur_params,
                                "vote":            np.int64(random_vote),
                            }
                            
                            #print(f"Action Prompt: Agent wählt {random_destination} für {random_duration} Steps")

                        env.step(action)
                end_time = time.perf_counter()
                runtime_in_ms = (end_time - start_time) * 1000

                print(f"{agents_amount},{episodes_amount},{locations_amount},{runtime_in_ms}\n")
                f.write(f"{agents_amount},{episodes_amount},{locations_amount},{runtime_in_ms}\n")
            
print("Test run complete!")
