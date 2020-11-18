# Runs a set of experiments defined in a json file in the experiments folder
# specify the json file to run by passing its name as a command line argument

import json
import time

from sys import argv
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit

from modules import load_data
from modules import experiment_objects

from modules.evaluate import write_results


DATA_DIR = Path.cwd() / "data"

# Get config values
print("Loading config values")
config_file = Path.cwd() / "configs/config.json"
with open(config_file, "r") as f:
    config = json.load(f)

# Load ratings data and trim down to selected values
print("Loading ratings data")
ratings = load_data.trim_ratings(
    load_data.ratings_data(), config["min_user_ratings"], config["min_book_ratings"]
)
ratings["recommend"] = load_data.set_threshold(ratings, config["threshold"])
ratings = load_data.set_class_proportions(
    ratings, config["min_proportion_positive"], config["max_proportion_positive"]
)
print(f"Data ready: {ratings.shape[0]} ratings to use")

# Train/test split - just get the indexes for now
sss = StratifiedShuffleSplit(
    n_splits=1,
    test_size=(1 - config["training_proportion"]),
    random_state=config["random_state"],
)
train, test = list(sss.split(ratings, ratings["recommend"]))[0]

print("Test/train split done")
print(f"{len(train)} ratings in training set")
print(f"{len(test)} ratings in test set")

# Create object to hold data
data = experiment_objects.ExperimentData(
    ratings, train, test, config["training_run"], config["random_state"], DATA_DIR
)

# Load the experiment configuration from json
print("Preparing experiments")
exp_file = argv[1]
if exp_file[-5:] != ".json":
    exp_file = exp_file + ".json"
experiments_file = Path.cwd() / "settings" / exp_file
with open(experiments_file, "r") as f:
    experiments = json.load(f)


# Run the experiments
for e in experiments:
    print(f"Now running: {e['description']}")
    start = time.time()
    running = experiment_objects.Experiment(data=data, **e)
    running.process_folds()

    if config["training_run"]:
        running.hybridise_folds()
        running.evaluate_folds()

    else:
        running.hybridise_test()
        running.retrain()
        running.evaluate_test()

    elapsed = time.time() - start
    write_results(config, running, elapsed)

    print(f"Completed {e['description']}")
    print(f"Time taken: {round(elapsed, 2)} seconds")
    print()

print("All experiments completed")
