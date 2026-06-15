from concurrent.futures import ProcessPoolExecutor
import time
import numpy as np
from swarm_environment.env.swarm_environment import SwarmDecisionEnvironment
import yaml


def run_single_setting(agents_amount, max_episodes_in_test, locations_amount, config):
    config["experiment"]["num_agents"] = agents_amount
    config["experiment"]["num_locations"] = locations_amount
    config["experiment"]["quorum_threshold"] = 2.0
    episodes_amount_testing = config["testing"]["episodes"]
    current_episode_max = 0

    start_time = time.perf_counter()
    lambdas = np.random.uniform(0.01, 1.0, locations_amount)
    env = SwarmDecisionEnvironment(config, lambdas)
    results = []

    for i in range(max_episodes_in_test):
        env.reset(lambdas=lambdas)

        while True:
            agent_id = env.agent_selection
            if agent_id is None or len(env.agents) == 0:
                break
            obs, reward, termination, truncation, info = env.last()

            if termination or truncation:
                action = None
            else:
                random_destination = np.random.randint(0, locations_amount + 1)
                random_dur_params = np.random.uniform(-3.0, 3.0, size=2).astype(
                    np.float32
                )
                random_vote = np.random.randint(0, locations_amount + 1)

                action = {
                    "location": np.int64(random_destination),
                    "duration_params": random_dur_params,
                    "vote": np.int64(random_vote),
                }

            env.step(action)
        lambdas = np.random.uniform(0.01, 1.0, locations_amount)

        if i == episodes_amount_testing[current_episode_max]:
            current_episode_max += 1
            end_time = time.perf_counter()
            runtime_int_ms = (end_time - start_time) * 1000
            print(
                f"{agents_amount},{episodes_amount_testing[current_episode_max]},{locations_amount},{runtime_int_ms}"
            )
            results.append(
                [
                    agents_amount,
                    episodes_amount_testing[current_episode_max],
                    locations_amount,
                    runtime_int_ms,
                ]
            )
    return results


if __name__ == "__main__":
    with open("config/configuration.yaml", "r") as f:
        config = yaml.safe_load(f)

    agents_amount_testing = config["testing"]["agents"]
    episodes_amount_testing = config["testing"]["episodes"]
    locations_amount_testing = config["testing"]["locations"]

    max_episodes_in_test = max(episodes_amount_testing)

    tasks = []
    for agents_amount in agents_amount_testing:
        for locations_amount in locations_amount_testing:
            tasks.append(
                (agents_amount, max_episodes_in_test, locations_amount, config)
            )

    print(
        f"Starte Parallelisierung für {len(tasks)} Kombinationen auf allen CPU-Kernen..."
    )

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_single_setting, *task) for task in tasks]
        results = [f.result() for f in futures]

    with open("tests/runtime_test.csv", "w") as f:
        f.write("agents, episodes, locations, runtime\n")
        for agents_amount, episodes_amount, locations_amount, runtime_in_ms in results:
            print(
                f"{agents_amount},{episodes_amount},{locations_amount},{runtime_in_ms}"
            )
            f.write(
                f"{agents_amount},{episodes_amount},{locations_amount},{runtime_in_ms}\n"
            )

    print("Test run complete!")
