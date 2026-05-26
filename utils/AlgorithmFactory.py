import random
import torch
from algorithms.IF import IsolationForest
from algorithms.ODIF import DeepIF

class AlgorithmFactory:
    def getAlgorithmFromName(algorithm_name):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if algorithm_name == "IsolationForest":
            return IsolationForest()
        elif algorithm_name == "OptimizedDeepIF":
            return DeepIF(optimization=True, device=device)
        elif algorithm_name == "DeepIF":
            return DeepIF(optimization=False, device=device)
        elif algorithm_name == 'ECOD':
            from pyod.models.ecod import ECOD
            model = ECOD()
            return model