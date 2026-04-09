
with open("config/configuration.yaml", "r") as f:
    config = yaml.safe_load(f)

experiment = Experiment(config)
experiment.run()