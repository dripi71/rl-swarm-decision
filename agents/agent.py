import numpy as np 

class Agent:
    def __init__(self, id, config):
        self.id = id
        self.next_action_duration = 0
        self.next_location = -1
        self.current_vote = None
        self.timesteps_at_location = [0 for _ in range(config["experiment"]["num_locations"])]
        self.events_at_location = [0 for _ in range(config["experiment"]["num_locations"])]
        self.confidence = [0 for _ in range(config["experiment"]["num_locations"])]

    def update_vote(self, num_locations):
        # if there are no timesteps at any location, the agent has no vote
        if sum(self.timesteps_at_location) == 0:
            self.current_vote = None
            return
        
        quality_estimates = []
        for i in range(num_locations):
            a = self.timesteps_at_location[i]
            b = self.events_at_location[i]
            quality_estimates.append(b / (a + 1))

        self.current_vote = int(np.argmax(quality_estimates))
           

