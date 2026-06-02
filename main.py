from experiments.experiment import Experiment
import ray
ray.init(num_cpus=64)

experiment = Experiment()
experiment.run()