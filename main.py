from experiments.experiment import Experiment
import ray
#ray.init(num_cpus=64)

ray.init()
experiment = Experiment()
experiment.run()