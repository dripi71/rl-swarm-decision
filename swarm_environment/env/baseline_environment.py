from numpy import random
import numpy as np
import logging

from priority_queue.priority_queue import Priority_Q
from config.constants import ActionTypes, QObjectIndices
from agents.bayesian_agent import BaselineAgent


class BaselineSimulation:
    def __init__(self, config: dict):
        self.config = config
        self.num_locations = config["experiment"]["num_locations"]
        self.nest_loc_index = self.num_locations  # convention: nest = last index
        self.max_steps = float(config["experiment"]["max_steps"])
        self.travel_time = config["experiment"]["travel_time"]
        self.quorum_threshold = config["experiment"]["quorum_threshold"]
        self.prior_a0 = config["experiment"]["prior_alpha_0"]
        self.prior_b0 = config["experiment"]["prior_beta_0"]

        # Populated in reset()
        self.lambdas: np.ndarray = np.array([])
        self.prio_Q: Priority_Q = Priority_Q()
        self.current_step: float = 0.0
        self.agent_objects: list[BaselineAgent] = []
        self.sampling_agents: list[list[BaselineAgent]] = []
        self.nesting_agents: list[BaselineAgent] = []
        self.experiment_best_location: int = 0
        self.swarm_decision: int | None = None

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            np.random.seed(seed)

        self.lambdas = self._generate_lambdas()
        self.experiment_best_location = int(np.argmin(self.lambdas))
        self.swarm_decision = None
        self.current_step = 0.0
        self.prio_Q = Priority_Q()
        self.agent_objects = []
        self.sampling_agents = [[] for _ in range(self.num_locations)]
        self.nesting_agents = []

        num_agents = self.config["experiment"]["num_agents"]
        for i in range(num_agents):
            agent = BaselineAgent(i, self.config, self.prior_a0, self.prior_b0)
            agent.next_location = self.nest_loc_index
            self.agent_objects.append(agent)
            self.nesting_agents.append(agent)
            # Initial random wait in Nest (see implementation of https://github.com/StudentWorkCPS/gamma_bots/blob/master/agent/my_agent.py)
            random_first_nest_wait = random.randint(0, 500)
            self.prio_Q.add([i, ActionTypes.NESTING_FINISHED, random_first_nest_wait])

        for loc in range(self.num_locations):
            delay = max(1, round(np.random.exponential(scale=1.0 / self.lambdas[loc])))
            self.prio_Q.add([loc, ActionTypes.LOCATION_EVENT, delay])

    def _generate_lambdas(self) -> np.ndarray:
        hardness = self.config["experiment"]["current_hardness"]
        red_lambda = (
            self.config["experiment"]["red_env_lambda_easy"]
            if hardness == "easy"
            else self.config["experiment"]["red_env_lambda_hard"]
        )
        blue_lambda = self.config["experiment"]["blue_env_lambda"]
        blue_left = np.random.randint(0, 2)
        return np.array(
            [blue_lambda, red_lambda] if blue_left else [red_lambda, blue_lambda]
        )

    def run_episode(self) -> dict:
        # Run a complete episode
        while True:
            event = self.prio_Q.pop()
            self.current_step = event[QObjectIndices.ACTIONTIME]

            if self.current_step >= self.max_steps:
                return self._metrics(outcome="truncated")

            self._dispatch(event)

            # Check quorum after every event
            decision = self._check_consensus()
            if decision is not None:
                self.swarm_decision = decision
                outcome = (
                    "correct" if decision == self.experiment_best_location else "wrong"
                )
                return self._metrics(outcome=outcome)

    def _dispatch(self, event: list) -> None:
        etype = event[QObjectIndices.EVENTTYPE]
        match etype:
            case ActionTypes.LOCATION_EVENT:
                self._on_location_event(event)
            case ActionTypes.SAMPLING:
                self._on_sampling(event)
            case ActionTypes.SAMPLING_FINISHED:
                self._on_sampling_finished(event)
            case ActionTypes.NESTING:
                self._on_nesting(event)
            case ActionTypes.NESTING_FINISHED:
                self._on_nesting_finished(event)

    def _on_location_event(self, event: list) -> None:
        loc = event[QObjectIndices.AGENTID]
        for agent in self.sampling_agents[loc]:
            agent.record_event(self.current_step)
            agent.events_at_location[loc] += 1
        # Schedule the next Poisson event at this location
        delay = max(1, round(np.random.exponential(scale=1.0 / self.lambdas[loc])))
        self.prio_Q.add([loc, ActionTypes.LOCATION_EVENT, self.current_step + delay])

    def _on_sampling(self, event: list) -> None:
        agent_id = event[QObjectIndices.AGENTID]
        agent = self.agent_objects[agent_id]
        loc = agent.next_location

        agent.begin_sampling(self.current_step)
        self.sampling_agents[loc].append(agent)

        agent.timesteps_at_location[loc] += agent.next_location_duration

        self.prio_Q.add(
            [
                agent_id,
                ActionTypes.SAMPLING_FINISHED,
                self.current_step + agent.next_location_duration,
            ]
        )

    def _on_sampling_finished(self, event: list) -> None:
        agent_id = event[QObjectIndices.AGENTID]
        agent = self.agent_objects[agent_id]
        loc = agent.next_location

        agent.update_belief(loc)
        agent.location_visits[loc] += 1
        agent.set_opinion()

        self.sampling_agents[loc].remove(agent)

        tt = self._travel_time(loc, self.nest_loc_index)
        agent.next_location = self.nest_loc_index
        self.prio_Q.add([agent_id, ActionTypes.NESTING, self.current_step + tt])

    def _on_nesting(self, event: list) -> None:
        agent_id = event[QObjectIndices.AGENTID]
        agent = self.agent_objects[agent_id]

        self.nesting_agents.append(agent)
        # Dmmd with belief sharing:
        # agent.communicate_in_nest(self.nesting_agents)

        # Compute dissemination time
        nest_duration = agent.compute_nest_time()
        agent.next_location_duration = nest_duration

        self.prio_Q.add(
            [agent_id, ActionTypes.NESTING_FINISHED, self.current_step + nest_duration]
        )

    def _on_nesting_finished(self, event: list) -> None:
        agent_id = event[QObjectIndices.AGENTID]
        agent = self.agent_objects[agent_id]

        opinions = [0] * self.num_locations
        for nesting_agent in self.nesting_agents:
            if nesting_agent.current_vote is not None:
                opinions[nesting_agent.current_vote] += 1

        if agent in self.nesting_agents:
            self.nesting_agents.remove(agent)

        # Choose next measurement location
        loc = agent.choose_location(opinions)
        agent.current_vote = loc
        obs_time = agent.compute_obs_time(loc)

        agent.next_location = loc
        agent.next_location_duration = obs_time

        tt = self._travel_time(self.nest_loc_index, loc)
        self.prio_Q.add([agent_id, ActionTypes.SAMPLING, self.current_step + tt])

    def _travel_time(self, from_loc: int, to_loc: int) -> int:
        if from_loc == to_loc:
            return 1
        return self.config["experiment"]["travel_time"]

    def _check_consensus(self) -> int | None:
        num_agents = self.config["experiment"]["num_agents"]
        required = num_agents * self.quorum_threshold
        votes: dict[int, int] = {}
        for agent in self.agent_objects:
            if agent.current_vote is not None:
                votes[agent.current_vote] = votes.get(agent.current_vote, 0) + 1
        for loc, count in votes.items():
            if count >= required:
                return loc
        return None

    def _metrics(self, outcome: str) -> dict:
        total_events = sum(sum(a.events_at_location) for a in self.agent_objects)
        final_votes: dict[int, int] = {}
        for agent in self.agent_objects:
            if agent.current_vote is not None:
                final_votes[agent.current_vote] = (
                    final_votes.get(agent.current_vote, 0) + 1
                )

        return {
            "outcome": outcome,
            "steps": self.current_step,
            "correct": self.swarm_decision == self.experiment_best_location
            if self.swarm_decision is not None
            else False,
            "truncated": outcome == "truncated",
            "total_events": total_events,
            "events_per_agent": total_events / len(self.agent_objects),
            "best_loc": self.experiment_best_location,
            "decision_loc": self.swarm_decision,
            "lambdas": list(self.lambdas),
            "final_votes": final_votes,
        }
