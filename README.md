# RL Swarm Decision Making

This repository provides a framework for collective decision-making in robot swarms using Multi-Agent Reinforcement Learning (MARL). The system utilizes PettingZoo AECEnv for simulating the swarm environment and Ray RLlib (PPO/MAPPO) for training and evaluating the agents.

---

## Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone git@github.com:dripi71/rl-swarm-decision.git
   cd rl-swarm-decision
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   * **Linux/macOS**:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
   Install the required packages defined in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration

All environment and training parameters are configured in [`config/configuration.yaml`].
Key configuration sections include:

* **`experiment`**:
  * `mode`: The run mode. Available options are:
    * `train`: Train a new policy or continue training an existing one.
    * `test`: Evaluate and compare a trained policy against the DMMD baseline.
    * `baseline`: Run simulations using only the Bayesian DMMD baseline.
  * `num_agents`: Number of robots in the swarm (e.g., `50`).
  * `num_locations`: Number of available locations/zones (e.g., `2`).
  * `test_run_name`: Name of the current run (used for policy checkpointing and log folders).
  * `load_checkpoint`: `true`/`false` – Controls whether to load an existing checkpoint during training or testing.
* **`rewards`**: Defines the reward structure for the RL agents (e.g., rewards for correct decisions, penalties for incorrect choices/timeouts, and bonuses for exploration and voting).

---

## Execution & Run Modes

Once you have configured the parameters in `config/configuration.yaml`, you can start the experiment run:

```bash
python main.py
```