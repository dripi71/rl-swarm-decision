from experiments.experiment import Experiment
import ray
ray.init(num_cpus=48)

experiment = Experiment()
experiment.run()