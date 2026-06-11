import numpy as np 
import random

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
        self.steps_at_current_location = 0
        self.uncertainties_before = np.ones(self.num_locations, dtype=np.float32)

        # For Bayesian Baseline
        #self.obs_time = 1000   # initial 10³ timesteps
        #self.location_visits = [0 for _ in range(self.num_locations)]
        #self.last_ten_visits_at_loc = [deque(maxlen=10) for _ in range(self.num_locations)]
        #self.event_counter = 0
        #self.initial_random_wait = random.randint(1, 500) # Reference : https://github.com/StudentWorkCPS/gamma_bots/blob/master/swarmy/experiment.py

    def update_vote(self):
        #todo: delete calls of update_vote
        pass
    
    def get_confidence(self):
        D = []
        for i in range(self.num_locations):
            d = self.events_at_location[i] / (self.timesteps_at_location[i] + 1)
            D.append(d)
        return D
           

