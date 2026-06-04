import numpy as np 

class Agent:
    def __init__(self, id, config):
        self.id = id
        self.next_action_duration = 0
        self.next_location_duration = 0
        self.next_location = -1
        self.current_vote = None
        self.num_locations = config["experiment"]["num_locations"]
        self.timesteps_at_location = [0 for _ in range(self.num_locations)]
        self.events_at_location = [0 for _ in range(self.num_locations)]
        self.confidence = [0 for _ in range(self.num_locations)]

    def update_vote(self):
        pass
           

