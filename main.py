import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import subprocess

from experiments.DatasetsSettings import DatasetsSettings

EXPERIMENTS_NUMBER = 2
algorithm_names = ["OptimizedDeepIF","DeepIF","IsolationForest"]
for dataset in DatasetsSettings.DATASETS_NAMES:
    for algorithm_name in algorithm_names:
        for experiment_id in range(EXPERIMENTS_NUMBER):
            subprocess.run([
                "python", "experiments/MemoryAndTimeExperiment.py",
                "--dataset_source", dataset["source"],
                "--algorithm_name", algorithm_name
            ])