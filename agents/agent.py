import numpy as np
import random


class Agent:
    def __init__(self, id, config):
        self.id = id
        self.next_location_duration = 0
        self.next_location = -1
        self.current_vote = None
        self.num_locations = config["experiment"]["num_locations"]
        self.timesteps_at_location = [0 for _ in range(self.num_locations)]
        self.events_at_location = [0 for _ in range(self.num_locations)]
        self.steps_at_current_location = 0
        self.uncertainties_before = np.ones(self.num_locations, dtype=np.float32)
        self.uncertainty_at_last_vote = np.ones(self.num_locations, dtype=np.float32)
