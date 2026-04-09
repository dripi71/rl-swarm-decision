class Agent:
    def __init__(self, id, config):
        self.id = id
        self.next_action_duration = 0
        self.next_action = -1
        self.current_vote = None
        self.timesteps_at_location = [0 for _ in range(config["experiment"]["num_locations"])]
        self.events_at_location = [0 for _ in range(config["experiment"]["num_locations"])]
        self.confidence = [0 for _ in range(config["experiment"]["num_locations"])]
