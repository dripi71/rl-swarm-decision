import numpy as np


class SwarmBase:
    """
    Base class for swarm decision environments for RL and baselines.
    Provides shared logic for lambda generation, consensus checking, and
    travel time calculation. Subclasses must ensure 'self.config' and
    'self.agent_objects' are set before calling any of these methods.
    """

    def generate_lambdas(self):
        mode = self.config["experiment"]["mode"]
        if mode == "train":
            use_easy = np.random.random() < 0.5
        else:
            use_easy = self.config["experiment"]["current_hardness"] == "easy"

        red_lambda = (
            self.config["experiment"]["red_env_lambda_easy"]
            if use_easy
            else self.config["experiment"]["red_env_lambda_hard"]
        )
        blue_lambda = self.config["experiment"]["blue_env_lambda"]

        num_locations = self.config["experiment"]["num_locations"]
        lambdas = np.full(num_locations, red_lambda, dtype=np.float64)
        good_index = np.random.randint(0, num_locations)
        lambdas[good_index] = blue_lambda
        return lambdas

    def check_consensus(self):
        quorum_threshold = self.config["experiment"]["quorum_threshold"]
        num_agents = self.config["experiment"]["num_agents"]
        required_votes = num_agents * quorum_threshold

        votes: dict[int, int] = {}
        for agent in self.agent_objects:
            if agent.current_vote is not None:
                votes[agent.current_vote] = votes.get(agent.current_vote, 0) + 1

        for loc, count in votes.items():
            if count >= required_votes:
                return loc
        return None

    def calculate_travel_time(self, location1, location2):
        if location1 == location2:
            return 1
        return self.config["experiment"]["travel_time"]
