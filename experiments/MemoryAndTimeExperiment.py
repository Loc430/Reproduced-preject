import argparse
import os
import gc
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import pynvml
import torch
import numpy as np

from experiments.DatasetsSettings import DatasetsSettings
from utils.AlgorithmFactory import AlgorithmFactory
from utils.MemoryMonitor import MemoryMonitor


def reset_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
def force_gc():
    gc.collect()

def runMemoryAndTimeExperiment(algorithm_name, dataset, xIn, yIn=None, max_warmup_samples = 1000):
    np.random.seed()

    algorithm = AlgorithmFactory.getAlgorithmFromName(algorithm_name)

    # CPU and GPU WARM-UP using at most the first max_warmup_samples samples
    warmup_limit = min(max_warmup_samples, len(xIn))
    algorithm.fit(xIn[:warmup_limit])

    algorithm_name = algorithm.algorithm_name()

    print(algorithm_name + " - " + dataset["name"] + ":")
    print("\tdimensionality = " + str(len(xIn[0])))
    print("\tsamples number = " + str(len(xIn)))

    reset_gpu_memory()
    force_gc()

    pid = os.getpid()
    monitor = MemoryMonitor(pid)
    monitor.start()

    algorithm.fit(xIn)

    monitor.stop()
    monitor.join()

    ram_usage = monitor.max_ram_usage
    vram_usage = monitor.max_vram_usage
    print(f"\tmax RAM usage: {float(ram_usage) / (1024 ** 2):.2f} MB")
    print(f"\tmax VRAM usage: {float(vram_usage) / (1024 ** 2):.2f} MB")
    print("\ttrain - cpu stage = " + str(algorithm.cpu_stages_fit_time / 1000000.0) + " [ms]")
    print("\ttrain - gpu stage = " + str(algorithm.gpu_stages_fit_time / 1000000.0) + " [ms]")

    reset_gpu_memory()
    force_gc()

pynvml.nvmlInit()

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_source", type=str, required=True)
parser.add_argument("--algorithm_name", type=str, required=True)
args = parser.parse_args()

dataset = [ds for ds in DatasetsSettings.DATASETS_NAMES if ds["source"] == args.dataset_source][0]

df = pd.read_csv('datasets/' + dataset["source"] + '.csv', sep=',', dtype=np.float64, engine='python')
dataXY = df.values
xIn = dataXY[:, :len(dataXY[0]) - 1]
yIn = dataXY[:, len(dataXY[0]) - 1:][:, 0]

runMemoryAndTimeExperiment(args.algorithm_name, dataset, xIn, yIn)

pynvml.nvmlShutdown()