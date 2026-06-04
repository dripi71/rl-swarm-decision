from pettingzoo import AECEnv
import gymnasium as gym
from priority_queue.priority_queue import Priority_Q
from config.constants import ActionTypes
from config.constants import PredictionKeys
from config.constants import QObjectIndices
import numpy as np
from agents.agent import Agent
import logging
import random


class SwarmDecisionEnvironment(AECEnv):
    metadata = {
        "name": "swarm_decision_v1",
    }

    def __init__(self, config):
        self.config = config
        self.lambdas = self.generateLambdas()
        self.sampling_agents = [[] for _ in range(self.config["experiment"]["num_locations"])]
        self.nesting_agents = []
        self.agents = []
        self.agent_objects = []
        self.locations = []
        self.prio_Q = Priority_Q()
        self.current_step = 0
        self.last_progress = 0.0
        self.swarm_decision = None
        self.experiment_best_location = np.argmin(self.lambdas)
        # The nest location index is the last index (num_locations because 0 is location 1)
        self.nest_loc_index = self.config["experiment"]["num_locations"]
        self.createLocationsAndAgents()

        self.rewards = { agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = { agent: False for agent in self.agents}
        self.truncations = { agent: False for agent in self.agents}
        self.infos = { agent: {} for agent in self.agents}

        self.action_spaces = {
            agent: gym.spaces.Dict({
                PredictionKeys.LOCATION: gym.spaces.Discrete(
                    self.config["experiment"]["num_locations"] + 1
                ),
                PredictionKeys.DURATION_PARAMS: gym.spaces.Box(
                    low=-10.0, high=10.0, shape=(2,), dtype=np.float32
                ),
                PredictionKeys.VOTE: gym.spaces.Discrete(
                    self.config["experiment"]["num_locations"] + 1
                ),
            }) for agent in self.agents
        }
        
        # observation space: loc_obs (N+1) + quality_estimates (N) + self_vote (N) + nest_votes_ratio (N) + Agent ID (break obs symmetrie) = 4*N + 2
        obs_dim = 4 * self.config["experiment"]["num_locations"] + 1 + 1
        self.observation_spaces = {
            agent: gym.spaces.Box(low=0.0, high=np.inf, shape=(obs_dim,), dtype=np.float32) 
            for agent in self.agents
        }
        
        self.possible_agents = self.agents[:]

        logging.basicConfig(filename="logs/last_run.txt", filemode="w", level=logging.INFO, format='%(message)s')

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.prio_Q = Priority_Q()
        self.current_step = 0
        self.nesting_agents = []
        self.agents = []
        self.agent_objects = []
        self.locations = []
        self.last_progress = 0.0
        self.swarm_decision = None
        self.sampling_agents = [[] for _ in range(self.config["experiment"]["num_locations"])]
        self.lambdas = self.generateLambdas()
        self.experiment_best_location = np.argmin(self.lambdas)
        self.createLocationsAndAgents()
        self.rewards = { agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = { agent: False for agent in self.agents}
        self.truncations = { agent: False for agent in self.agents}
        self.infos = { agent: {} for agent in self.agents}

    def generateLambdas(self):

        if(self.config["experiment"]["current_hardness"] == "easy"):
            red_loc_lambda = self.config["experiment"]["red_env_lambda_easy"]
        else:
            red_loc_lambda = self.config["experiment"]["red_env_lambda_hard"]
        
        blue_loc_lambda = self.config["experiment"]["blue_env_lambda"]

        blue_left = np.random.randint(0, 2)
        if(blue_left):
            return np.array([blue_loc_lambda, red_loc_lambda])
        else:
            return np.array([red_loc_lambda, blue_loc_lambda])
    

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        agent_id = self.agent_selection
        self._cumulative_rewards[agent_id] = 0
        self._clear_rewards()

        agent = self.get_agent_by_id(agent_id)
        current_location = agent.next_location

        if(current_location == self.nest_loc_index):
            # agent was in Nest
            self.nesting_agents.remove(agent)
            # agent takes beliefs of other nesting agents into account
            self.influence_agent(agent)

        elif current_location in self.locations:
            self.sampling_agents[current_location].remove(agent)
            # agent is finished with sampling -> update beliefs based on collected evidence

        if(self.swarm_reached_decision()):
            return

        # agent has to pay cost for making a decision
        # otherwise: NN spams decisions (micro steps)
        self.rewards[agent_id] += self.config["rewards"]["decision_cost"]

        agent.next_location = action[PredictionKeys.LOCATION]
        agent.next_location_duration = self._sample_gamma_duration(
            action[PredictionKeys.DURATION_PARAMS]
        )
        
        # Parse vote action
        vote_action = action[PredictionKeys.VOTE]

        # Agent has to see at least one event at each location
        # -> prevents NN from blind guessing

        if vote_action == 0 or 0 in agent.events_at_location:
            agent.current_vote = None
        else:
            agent.current_vote = int(vote_action - 1)
        traveltime = self.calculate_travel_time(current_location, agent.next_location)

        if(agent.next_location == self.nest_loc_index):
            self.prio_Q.add([agent.id, ActionTypes.NESTING, self.current_step + traveltime])
        else:
            self.prio_Q.add([agent.id, ActionTypes.SAMPLING, self.current_step + traveltime])


        ## Reward system
        next_event = self.prio_Q.pop()
        while(not self.needs_prediction(next_event)):
            self.process_deterministic_event(next_event)
            # after each deterministic event, check if swarm has reached a decision
            if(self.swarm_reached_decision()):
                return
            next_event = self.prio_Q.pop()

        # Next event that needs a prediction
        self.agent_selection = next_event[QObjectIndices.AGENTID]
        self.current_step = next_event[QObjectIndices.ACTIONTIME]

        
        num_agents = len(self.agents)
        votes_for_best_loc = sum( 1 for a in self.agent_objects if a.current_vote == self.experiment_best_location)
        progress = votes_for_best_loc / num_agents

        # only give progess bonus if it improves -> prevent agent from milking progress rewards with stale progess

        if progress > self.last_progress:
            for agent_id in self.agents:
                agent = self.get_agent_by_id(agent_id)
                if agent.current_vote == self.experiment_best_location:
                    self.rewards[agent_id] += (progress - self.last_progress) * self.config["rewards"]["progress_bonus"]
            self.last_progress = progress


        if self.current_step >= float(self.config["experiment"]["max_steps"]):
            for agent_id in self.agents:
                self.rewards[agent_id] += self.config["rewards"]["reward_for_wrong_decision"]
            self.truncations = {agent: True for agent in self.agents}
            logging.info(f"Maximum steps reached, truncated: {self.current_step}")
            self._accumulate_rewards()
            return

        self._accumulate_rewards()
                
        
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def observe(self, agent_id):

        agent = self.get_agent_by_id(agent_id)
        num_locations = self.config["experiment"]["num_locations"]
        # one hot encoding for current location of agent
        loc_obs = np.zeros((num_locations + 1), dtype=np.float32)
        loc_obs[agent.next_location] = 1.0
        quality_estimates = np.zeros(num_locations, dtype=np.float32)
        for i in range(num_locations):
            a = agent.timesteps_at_location[i]
            b = agent.events_at_location[i]
            quality_estimates[i] = b / (a + 1)
        self_vote = np.zeros(num_locations, dtype=np.float32)
        if (agent.current_vote is not None):
            self_vote[agent.current_vote] = 1.0
        nest_votes_ratio = np.zeros(num_locations, dtype=np.float32)
        # if the agent is currently in the nest, show opinions of other nesting agents
        if (agent.next_location == self.nest_loc_index):
            total_nest_agents = len(self.nesting_agents)
            if total_nest_agents > 0:
                for nest_agent in self.nesting_agents:
                    if nest_agent.current_vote is not None:
                        nest_votes_ratio[nest_agent.current_vote] += 1
                nest_votes_ratio = nest_votes_ratio / total_nest_agents
        
        # Normalized agent ID to break observation symmetry between agents
        num_agents = self.config["experiment"]["num_agents"]
        agent_id_norm = np.array([agent.id / max(num_agents - 1, 1)], dtype=np.float32)
        observation = np.concatenate([loc_obs, quality_estimates, self_vote, nest_votes_ratio, agent_id_norm])
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
            delay = max(1, round(np.random.exponential(scale=1.0 / self.lambdas[i])))
            self.prio_Q.add([i, ActionTypes.LOCATION_EVENT, self.current_step + delay])
        
        # Pop the first event to set the initial agent selection
        first_event = self.prio_Q.pop()
        self.agent_selection = first_event[QObjectIndices.AGENTID]
        self.current_step = first_event[QObjectIndices.ACTIONTIME]


    def _softplus(self, x):
        return np.log1p(np.exp(np.clip(x, -30, 30)))

    def _sample_gamma_duration(self, duration_params):
        epsilon = self.config["experiment"]["gamma_epsilon"]
        x1, x2 = float(duration_params[0]), float(duration_params[1])
        alpha = self._softplus(x1) + epsilon
        beta  = self._softplus(x2) + epsilon
        sample = np.random.gamma(shape=alpha, scale=1.0 / beta)
        return max(1, round(float(sample)))

    def render(self):
        pass

    def close(self):
        pass

    def needs_prediction(self, event):
        # The agent only needs a prediciton if it finished nesting or sampling
        # otherwise the next step(s) are deterministic (e.g. traveling to the location)
        return event[QObjectIndices.EVENTTYPE] in [ActionTypes.NESTING_FINISHED, ActionTypes.SAMPLING_FINISHED]
    
    def calculate_travel_time(self, location1, location2):
        if(location1 == location2):
            # Takes one step in order to prevent freezing time
            return 1
        else:
            return self.config["experiment"]["travel_time"]

    def get_agent_by_id(self, id):
        return self.agent_objects[id]

    def add_agent_to_nest(self, id):
        agent = self.get_agent_by_id(id)
        self.nesting_agents.append(agent)
    def add_agent_to_sampling(self, id):
        agent = self.get_agent_by_id(id)
        self.sampling_agents[agent.next_location].append(agent)

    def process_deterministic_event(self, event):

        self.current_step = event[QObjectIndices.ACTIONTIME]
        current_agent_id = event[QObjectIndices.AGENTID]

        match(event[QObjectIndices.EVENTTYPE]):
            case ActionTypes.NESTING:
                self.add_agent_to_nest(current_agent_id)
                next_location_duration = self.get_agent_by_id(current_agent_id).next_location_duration
                nextEvent = [current_agent_id, ActionTypes.NESTING_FINISHED, self.current_step + next_location_duration]
                self.prio_Q.add(nextEvent)
            case ActionTypes.SAMPLING:
                self.add_agent_to_sampling(current_agent_id)
                agent = self.get_agent_by_id(current_agent_id)
                next_location_duration = agent.next_location_duration
                agent.timesteps_at_location[agent.next_location] += next_location_duration
                nextEvent = [current_agent_id, ActionTypes.SAMPLING_FINISHED, self.current_step + next_location_duration]
                self.prio_Q.add(nextEvent)
            case ActionTypes.LOCATION_EVENT:
                location_id = event[QObjectIndices.AGENTID]
                for agent in self.sampling_agents[location_id]:
                    agent.events_at_location[location_id] += 1
                    self.rewards[agent.id] += self.config["rewards"]["reward_per_event"]
                    # information update -> update belief
                    agent.update_vote()
                delay = max(1, round(np.random.exponential(scale=1.0 / self.lambdas[location_id])))
                self.prio_Q.add([location_id, ActionTypes.LOCATION_EVENT, self.current_step + delay])

    def state(self):
        num_locations = self.config["experiment"]["num_locations"]
        num_agents = self.config["experiment"]["num_agents"]
        
        agent_counts = np.zeros(num_locations + 1, dtype=np.float32)
        for i in range(num_locations):
            agent_counts[i] = len(self.sampling_agents[i]) / num_agents
        agent_counts[self.nest_loc_index] = len(self.nesting_agents) / num_agents
        
        true_lambdas = np.array(self.lambdas, dtype=np.float32)
        
        global_state = np.concatenate([agent_counts, true_lambdas])
        return global_state

    def check_consensus(self):
        quorum_threshold = self.config["experiment"]["quorum_threshold"]
        num_agents = self.config["experiment"]["num_agents"]
        required_votes = num_agents * quorum_threshold
        votes = {loc: 0 for loc in range(self.config["experiment"]["num_locations"])}
        
        # Consensus is checked globally
        for agent in self.agent_objects:
            if agent.current_vote is not None:
                votes[agent.current_vote] += 1
        
        # there must be enough votes (quorum) to make a decision
        for loc, vote_count in votes.items():
            if vote_count >= required_votes:
                # return the location where the required votes are met
                return loc
        return None
    
    def influence_agent(self, agent):
        # DMMD
        if len(self.nesting_agents) == 0:
            return
            
        for location in range(self.config["experiment"]["num_locations"]):
            a = agent.timesteps_at_location[location]
            b = agent.events_at_location[location]

            a_others = sum(nesting_agent.timesteps_at_location[location] for nesting_agent in self.nesting_agents)
            b_others = sum(nesting_agent.events_at_location[location] for nesting_agent in self.nesting_agents)
            
            a_others_avg = a_others/len(self.nesting_agents)
            b_others_avg = b_others/len(self.nesting_agents)

            agent.timesteps_at_location[location] = (a + a_others_avg) / 2
            agent.events_at_location[location] = (b + b_others_avg) / 2
        
        agent.update_vote()
    def swarm_reached_decision(self):
        self.swarm_decision = self.check_consensus()
        if self.swarm_decision is not None:
            if self.swarm_decision == self.experiment_best_location:
                for agent_id in self.agents:
                    decayed_reward = self.config["rewards"]["reward_for_correct_decision"] - self.current_step * float(self.config["rewards"]["solved_bonus_time_decay"])
                    self.rewards[agent_id] += max(0.0, decayed_reward)
            else:
                for agent_id in self.agents:
                    self.rewards[agent_id] += self.config["rewards"]["reward_for_wrong_decision"]
            self.terminations = { agent: True for agent in self.agents}
            self._accumulate_rewards()
            return True
        return False
