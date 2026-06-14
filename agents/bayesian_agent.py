import numpy as np
from scipy.stats import chi2
import random

class BaselineAgent:
    """
    Bayesian agent implementing the DMMD algorithm from the paper.

    Belief model: inverse-gamma IG(a, b) as conjugate prior for
    exponential interarrival times (Weibull with beta=1).
      - a: number of observed events (shape)
      - b: sum of observed interarrival times (scale)
    Bayesian update: a' = a + n,  b' = b + sum(xi)

    Timing:
      tdiss = 500 / (|CI| + 0.2),  capped at 2500
      tobs  = 2 * ceil(CI_upper)
    """

    # Paper constants
    C_DISS = 500
    T_DISS_MAX = 2500

    def __init__(self, agent_id: int, config: dict, prior_a0: float = 1.0, prior_b0: float = 10000.0):
        self.id = agent_id
        self.num_locations = config["experiment"]["num_locations"]

        # --- Bayesian parameters per location ---
        self.a = np.full(self.num_locations, prior_a0, dtype=np.float64)
        self.b = np.full(self.num_locations, prior_b0, dtype=np.float64)

        # --- State for the finite-state machine ---
        self.current_vote: int | None = None
        self.next_location: int = config["experiment"]["num_locations"]  # start in nest
        self.next_location_duration: int = 0

        # Mirrors of environment counters (for compatibility with env observation)
        self.events_at_location = [0 for _ in range(self.num_locations)]
        self.timesteps_at_location = [0 for _ in range(self.num_locations)]
        self.steps_at_current_location: int = 0
        self.uncertainties_before = np.ones(self.num_locations, dtype=np.float32)

        # --- Interarrival time tracking (per sampling visit) ---
        self.interarrival_buffer: list[float] = []
        self.last_event_step: float = 0.0   # step of last event (or arrival)

        # --- Visit tracking (for location selection balancing) ---
        self.location_visits = [0 for _ in range(self.num_locations)]

        # Received data while nesting
        self.received_opinions: list[int] = [0 for _ in range(self.num_locations - 1)]
        

    def compute_ci(self, loc: int) -> tuple[float, float, float]:
        a = max(self.a[loc], 1e-6)
        b = max(self.b[loc], 1e-6)
        df = 2.0 * a
        lower = 2.0 * b / chi2.ppf(0.975, df)
        upper = 2.0 * b / chi2.ppf(0.025, df)
        width = upper - lower
        return lower, upper, width

    def compute_nest_time(self) -> int:
        # Paper: tdiss = C_diss / (max_CI_width + 0.2), capped at max_diss = 2500
        max_width = max(self.compute_ci(loc)[2] for loc in range(self.num_locations))
        tdiss = self.C_DISS / (max_width + 0.2)
        return int(min(self.T_DISS_MAX, max(1, round(tdiss))))

    def compute_obs_time(self, loc: int) -> int:
        #tobs = 2 * ceil(CI_upper) for the given location
        _, upper, _ = self.compute_ci(loc)
        return max(1, 2 * int(np.ceil(upper)))

    def update_belief(self, loc: int) -> None:
        times = self.interarrival_buffer
        n = len(times)
        if n > 0:
            self.a[loc] += n
            self.b[loc] += float(np.sum(times))
        self.interarrival_buffer = []

    def record_event(self, current_step: float) -> None:
        iat = current_step - self.last_event_step
        self.interarrival_buffer.append(iat)
        self.last_event_step = current_step

    def begin_sampling(self, current_step: float) -> None:
        # measure the interarrival time either since last event or sampling start
        self.last_event_step = current_step
        self.interarrival_buffer = []


    def set_opinion(self) -> None:
        if self.num_locations < 2:
            return

        ci_bounds = [self.compute_ci(loc) for loc in range(self.num_locations)]

        # Safer area = higher interarrival time = lower lambda
        modes = [self.b[loc] / (self.a[loc] + 1) for loc in range(self.num_locations)]
        best_loc = int(np.argmax(modes))
        best_lower = ci_bounds[best_loc][0]

        separated = all(
            ci_bounds[other][1] < best_lower
            for other in range(self.num_locations)
            if other != best_loc
        )

        if separated:
            self.current_vote = best_loc
        else:
            self.current_vote = None 

    def communicate_in_nest(self, nesting_agents: list) -> None:
        others = [ag for ag in nesting_agents if ag.id != self.id]
        if not others:
            return

        a_others = np.mean([ag.a for ag in others], axis=0)
        b_others = np.mean([ag.b for ag in others], axis=0)

        self.a = (self.a + a_others) / 2.0
        self.b = (self.b + b_others) / 2.0

    def choose_location(self, opinions: list[int]) -> int:
        next_loc = np.max(opinions)
        # if tie, then randomly chose
        best_locations =[loc for loc, votes in enumerate(opinions) if votes == next_loc]
        next_loc = random.choice(best_locations)
        return next_loc
