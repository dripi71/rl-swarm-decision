from pettingzoo import AECEnv
import gymnasium as gym
from priority_queue.priority_queue import Priority_Q
from config.constants import ActionTypes
from config.constants import PredictionIndices
from config.constants import QObjectIndices
import numpy as np
from agents.agent import Agent

class SwarmDecisionEnvironment(AECEnv):
    metadata = {
        "name": "swarm_decision_v1",
    }

    def __init__(self, config, lambdas):
        self.config = config
        self.lambdas = lambdas
        self.sampling_agents = [[] for _ in range(self.config["experiment"]["num_locations"])]
        self.nesting_agents = []
        self.agents = []
        self.agent_objects = []
        self.locations = []
        self.prio_Q = Priority_Q()
        self.current_step = 0
        # The nest location index is the last index (num_locations because 0 is location 1)
        self.nest_loc_index = self.config["experiment"]["num_locations"]
        # action space: 1. Variable: 0 - num_locations: wohin soll als nächstes gegangen werden loc1, loc2, ..., locN, NEST
        #               2. Variable: 0 - max_wait: wie lange soll gewartet werden
        self.action_spaces = gym.spaces.Discrete(self.config["experiment"]["num_locations"] + 1)
        # observation space: num_locations + nest + confidence + votes

        # muss noch angepasst werden
        self.observation_spaces = gym.spaces.Box(low=0, high=1, shape=(self.config["experiment"]["num_locations"] + 1,))
        
        self.createLocationsAndAgents()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.prio_Q = Priority_Q()
        self.current_step = 0
        self.nesting_agents = []
        self.agents = []
        self.agent_objects = []
        self.locations = []
        self.sampling_agents = [[] for _ in range(self.config["experiment"]["num_locations"])]
        self.createLocationsAndAgents()

    def step(self, action):

        # [AgentId, EventType, ActionTime]        
        agent_id = self.agent_selection
        agent = self.get_agent_by_id(agent_id)
        # Skip world time forward to the next event
        self.current_step = agent[QObjectIndices.ACTIONTIME]

        if(agent.nextDestination == self.nest_loc_index):
            self.nesting_agents.remove(agent)
        elif agent.nextDestination in self.locations:
            self.sampling_agents[agent.nextDestination].remove(agent)

        agent.next_action = action[PredictionIndices.LOCATION]
        # Predicted stay time for the next location
        agent.next_action_duration = action[PredictionIndices.DURATION]

        traveltime = self.calculate_travel_time(agent.currentLocation, agent.nextDestination)

        if(agent.nextDestination == self.nest_loc_index):
            self.prio_Q.add([agent.id, ActionTypes.NESTING, self.current_step + traveltime])
        else:
            self.prio_Q.add([agent.id, ActionTypes.SAMPLING, self.current_step + traveltime])


        # Hier kommt dann das reward system rein


        next_event = self.prio_Q.pop()
        while(not self.needs_prediction(next_event[QObjectIndices.EVENTTYPE])):
            self.current_step = next_event[QObjectIndices.ACTIONTIME]
            self.process_deterministic_event(next_event)
            next_event = self.prio_Q.pop()

        # Next event that needs a prediction
        self.agent_selection = next_event[QObjectIndices.AGENTID]
        return self.get_observation(self.agent_selection)
        
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def get_observation(self, agent_id):
        agent = self.get_agent_by_id(agent_id)
        num_locations = self.config["experiment"]["num_locations"]
        loc_obs = np.zeros((num_locations + 1), dtype=np.float32)
        loc_obs[agent.nextDestination] = 1.0
        quality_estimates = np.zeros(num_locations, dtype=np.float32)
        for i in range(num_locations):
            a = 1.0 + agent.timesteps_at_location[i]
            b = 1.0 + agent.events_at_location[i]
            quality_estimates[i] = a / (a + b)
        self_vote = np.zeros(num_locations, dtype=np.float32)
        if (agent.current_vote is not None):
            self_vote[agent.current_vote] = 1.0
        nest_votes_ratio = np.zeros(num_locations, dtype=np.float32)
        if( agent.nextDestination == self.nest_loc_index):
            total_nest_agents = len(self.nesting_agents)
            if total_nest_agents > 0:
                for nest_agent in self.nesting_agents:
                    if nest_agent.current_vote is not None:
                        nest_votes_ratio[nest_agent.current_vote] += 1
                nest_votes_ratio = nest_votes_ratio / total_nest_agents
        
        observation = np.concatenate([loc_obs, quality_estimates, self_vote, nest_votes_ratio])
        return observation

    def createLocationsAndAgents(self):
        for i in range(self.config["experiment"]["num_agents"]):
            # agents is reserved by pettingzoo, only list ids here
            self.agents.append(i)
            # create own agent object list
            self.agent_objects.append(Agent(i, self.config))
            self.prio_Q.add([i, ActionTypes.NESTING_FINISHED, 0])
        
        for i in range(self.config["experiment"]["num_locations"]):
            self.locations.append(i)
            delay = np.random.exponential(scale=1.0 / self.lambdas[i])
            self.prio_Q.add([i, ActionTypes.LOCATION_EVENT, self.current_step + delay])
        
        # Pop the first event to set the initial agent selection
        first_event = self.prio_Q.pop()
        self.agent_selection = first_event[QObjectIndices.AGENTID]


    def render(self):
        pass

    def needs_prediction(self, agent):
        return agent[QObjectIndices.EVENTTYPE] in [ActionTypes.NESTING_FINISHED, ActionTypes.SAMPLING_FINISHED]
    
    def calculate_travel_time(self, location1, location2):
        #First for simplicity: everything has an equal distance
        if(location1 == location2):
            return 0
        else:
            return self.config["experiment"]["travel_time"]

    def get_agent_by_id(self, id):
        return self.agent_objects[id]

    def add_agent_to_nest(self, id):
        agent = self.get_agent_by_id(id)
        self.nesting_agents.append(agent)
    def add_agent_to_sampling(self, id):
        agent = self.get_agent_by_id(id)
        self.sampling_agents[agent.nextDestination].append(agent)

    def process_deterministic_event(self, Qevent):

        current_agent_id = Qevent[QObjectIndices.AGENTID]

        match(Qevent[QObjectIndices.EVENTTYPE]):
            case ActionTypes.NESTING:
                self.add_agent_to_nest(current_agent_id)
                next_action_duration = self.get_agent_by_id(current_agent_id).next_action_duration
                nextEvent = [current_agent_id, ActionTypes.NESTING_FINISHED, self.current_step + next_action_duration]
                self.prio_Q.add(nextEvent)
            case ActionTypes.SAMPLING:
                self.add_agent_to_sampling(current_agent_id)
                agent = self.get_agent_by_id(current_agent_id)
                next_action_duration = agent.next_action_duration
                agent.timesteps_at_location[agent.nextDestination] += next_action_duration
                nextEvent = [current_agent_id, ActionTypes.SAMPLING_FINISHED, self.current_step + next_action_duration]
                self.prio_Q.add(nextEvent)
            case ActionTypes.LOCATION_EVENT:
                location_id = Qevent[QObjectIndices.AGENTID]
                for agent in self.sampling_agents[location_id]:
                    agent.events_at_location[location_id] += 1
                delay = np.random.exponential(scale=1.0 / self.lambdas[location_id])
                self.prio_Q.add([location_id, ActionTypes.LOCATION_EVENT, self.current_step + delay])