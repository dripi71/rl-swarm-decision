from experiments.experiment import Experiment
import ray
import yaml

with open("config/configuration.yaml", "r") as f:
    config = yaml.safe_load(f)

num_cpus = config["hardware"]["num_cpus"]

ray.init(num_cpus=num_cpus)
experiment = Experiment()
experiment.run()