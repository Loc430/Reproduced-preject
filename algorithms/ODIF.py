import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.init
# Flow A — Normalization (#4)
from sklearn.preprocessing import StandardScaler

from algorithms.IF import IsolationForest, IsolationTree, Node

TREE_SEED_HIGH = np.iinfo(np.int32).max

try:
    from networks.MLPNetwork import MLPNetwork
except ModuleNotFoundError:
    class MLPNetwork(torch.nn.Module):
        def __init__(self, n_features, network_hidden_dimensions, representation_dimensionality,
                     representations_number, activation_fun='tanh', device='cuda'):
            super().__init__()
            activation_layer = torch.nn.Tanh if activation_fun == 'tanh' else torch.nn.ReLU
            layers = []
            input_dim = n_features
            for hidden_dim in network_hidden_dimensions:
                layers.append(torch.nn.Linear(input_dim, hidden_dim))
                layers.append(activation_layer())
                input_dim = hidden_dim
            layers.append(torch.nn.Linear(
                input_dim, representation_dimensionality * representations_number))
            self.model = torch.nn.Sequential(*layers)
            self.to(device)

        def forward(self, x):
            return self.model(x)


def _build_tree_node(samples, depth, max_depth, rng):
    if len(samples) <= 1 or depth >= max_depth:
        return Node(samples_number=len(samples))

    feature = rng.randint(0, samples.shape[1])
    min_val = np.min(samples[:, feature])
    max_val = np.max(samples[:, feature])
    threshold = rng.uniform(min_val, max_val)

    left_mask = samples[:, feature] < threshold
    right_mask = ~left_mask

    left_node = _build_tree_node(samples[left_mask], depth + 1, max_depth, rng)
    right_node = _build_tree_node(samples[right_mask], depth + 1, max_depth, rng)
    return Node(feature, threshold, left_node, right_node)


def build_one_tree(tree_args):
    samples, max_depth, seed = tree_args
    tree = IsolationTree(max_depth)
    rng = np.random.RandomState(seed)
    tree.root = _build_tree_node(samples, depth=0, max_depth=max_depth, rng=rng)
    return tree


class DeepIF:
    def __init__(self, optimization=True, representations_number=50, trees_per_representation=6,
                 samples_number_per_tree=256,
                 network_hidden_dimensions=[500, 100], representation_dimensionality=20, batch_size=64,
                 device='cuda', seed=None):

        self.representations_number = representations_number
        self.trees_per_representation = trees_per_representation
        self.samples_number_per_tree = samples_number_per_tree
        self.network_hidden_dimensions = network_hidden_dimensions
        self.representation_dimensionality = representation_dimensionality
        self.batch_size = batch_size
        self.device = device
        self.optimization = optimization
        self.cpu_stages_fit_time = None
        self.gpu_stages_fit_time = None
        self.threadpool_fallback_used = False
        # Flow A — scaler fitted during fit(), applied in decision_function()
        self.scaler = None

        if seed:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        else:
            np.random.seed(random.randint(0, 2 ** 32 - 1))
            torch.manual_seed(random.randint(0, 2 ** 32 - 1))
            torch.cuda.manual_seed_all(random.randint(0, 2 ** 32 - 1))
        return

    def _reshape_batch_representation(self, batch_representation, batch_size):
        # FIX #1 (bug from merge): restore original reshape semantics.
        # MLP outputs flat [B, R*F]; original paper code reshaped to [R, B, F]
        # (the "scrambled" view that gave high ensemble diversity).
        # The merged version's reshape(B, R, F) collapses diversity and was
        # responsible for the ~-29% PR-AUC regression.
        if batch_representation.dim() == 3:
            if batch_representation.shape[0] == self.representations_number and batch_representation.shape[1] == batch_size:
                return batch_representation.contiguous()
            if batch_representation.shape[0] == batch_size and batch_representation.shape[1] == self.representations_number:
                return batch_representation.permute(1, 0, 2).contiguous()
        return batch_representation.reshape(self.representations_number, batch_size, -1)

    # Flow B — batched inference with torch.no_grad() (#6)
    def _collect_representations(self, X):
        # FIX #1: per-batch shape is now [R, B_i, F]; concatenate along the
        # batch axis (dim=1) so the final tensor is [R, N, F].
        batch_size = getattr(self, 'batch_size', 256) or 256
        all_reps = []
        with torch.no_grad():
            for start_idx in range(0, len(X), batch_size):
                batch = torch.as_tensor(X[start_idx:start_idx + batch_size], dtype=torch.float32, device=self.device)
                batch_representation = self.network(batch)
                batch_representation = self._reshape_batch_representation(
                    batch_representation, batch.shape[0]).detach().cpu()
                all_reps.append(batch_representation)
        return torch.cat(all_reps, dim=1)

    def _create_empty_isolation_forest(self):
        np_state = np.random.get_state()
        try:
            return IsolationForest(trees_number=self.trees_per_representation,
                                   samples_number=self.samples_number_per_tree, seed=1)
        finally:
            np.random.set_state(np_state)

    # Flow B — parallel tree construction with ThreadPoolExecutor (#7)
    def _build_trees(self, tree_args_list):
        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                trees = list(executor.map(build_one_tree, tree_args_list))
        except Exception:
            self.threadpool_fallback_used = True
            trees = [build_one_tree(tree_args) for tree_args in tree_args_list]
        return trees

    def fit(self, X, Y=None):
        # FIX #2: removed unconditional StandardScaler.fit_transform here.
        # Normalization belongs to the caller (run_flowa.py --normalize),
        # not the model. The previous code re-normalized even when the
        # caller had explicitly opted out.
        self.scaler = None

        cpu_stage_1_start_time = time.perf_counter_ns()
        self.threadpool_fallback_used = False

        self.n_features = X.shape[-1]
        self.network = MLPNetwork(n_features=self.n_features, network_hidden_dimensions=self.network_hidden_dimensions,
                                  representation_dimensionality=self.representation_dimensionality,
                                  representations_number=self.representations_number, activation_fun='tanh',
                                  device=self.device)
        for name, parameter in self.network.named_parameters():
            if name.endswith('weight'):
                torch.nn.init.normal_(parameter, mean=0.0, std=1.0)
        self.network.eval()

        if self.optimization == False:
            cpu_stage_1_stop_time = time.perf_counter_ns()

            representations = self._collect_representations(X)

            cpu_stage_2_start_time = time.perf_counter_ns()

            # FIX #1: representations is now [R, N, F]; iterate on dim 0
            x_representation_list = [representations[i, :, :].numpy() for i in range(representations.shape[0])]

            self.isolation_forest_list = []
            for i in range(self.representations_number):
                i_forest = self._create_empty_isolation_forest()
                tree_args_list = []
                n_samples = x_representation_list[i].shape[0]
                tree_seeds = np.random.randint(0, TREE_SEED_HIGH, size=self.trees_per_representation)
                for j in range(self.trees_per_representation):
                    indices = np.random.choice(n_samples, self.samples_number_per_tree, replace=True)
                    tree_args_list.append((
                        x_representation_list[i][indices], i_forest.max_depth, int(tree_seeds[j])))
                i_forest.trees = self._build_trees(tree_args_list)
                self.isolation_forest_list.append(i_forest)
            cpu_stage_2_stop_time = time.perf_counter_ns()
        else:
            sampled_indices = np.random.choice(len(X), self.samples_number_per_tree * self.trees_per_representation,
                                               replace=True)
            unique_indices, inverse_indices = np.unique(sampled_indices, return_inverse=True)

            cpu_stage_1_stop_time = time.perf_counter_ns()

            representations = self._collect_representations(X[unique_indices])

            cpu_stage_2_start_time = time.perf_counter_ns()

            # FIX #1: representations is now [R, N, F]; iterate on dim 0
            x_representation_list = [representations[i, :, :].numpy() for i in range(representations.shape[0])]

            self.isolation_forest_list = []
            for i in range(self.representations_number):
                i_forest = self._create_empty_isolation_forest()
                tree_args_list = []
                tree_seeds = np.random.randint(0, TREE_SEED_HIGH, size=self.trees_per_representation)
                for j in range(self.trees_per_representation):
                    tree_args_list.append((
                        x_representation_list[i][inverse_indices[
                            j * self.samples_number_per_tree:(j + 1) * self.samples_number_per_tree]],
                        i_forest.max_depth,
                        int(tree_seeds[j])
                    ))
                i_forest.trees = self._build_trees(tree_args_list)
                self.isolation_forest_list.append(i_forest)
            cpu_stage_2_stop_time = time.perf_counter_ns()

        cpu_stages_elapsed_time = cpu_stage_1_stop_time - cpu_stage_1_start_time + cpu_stage_2_stop_time - cpu_stage_2_start_time
        gpu_stages_elapsed_time = cpu_stage_2_start_time - cpu_stage_1_stop_time
        self.cpu_stages_fit_time = cpu_stages_elapsed_time
        self.gpu_stages_fit_time = gpu_stages_elapsed_time

    def decision_function(self, X, aggregation='mean'):
        """
        Compute anomaly scores.

        Flow A — Score Aggregation (#3):
          aggregation='mean'     : average across all R representations (original behaviour)
          aggregation='trimmed'  : mean after dropping the top and bottom 10% of per-rep scores
          aggregation='weighted' : weight each representation by its score variance
                                   (higher-variance reps contribute more)

        Flow B — batched no_grad inference is handled by _collect_representations().

        Parameters
        ----------
        X : np.ndarray
        aggregation : str, default 'mean'

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        # FIX #2: scaler is intentionally None now; caller controls
        # normalization. Kept the attribute for backwards-compat.
        if self.scaler is not None:
            X = self.scaler.transform(X)

        # Flow B — batched, no_grad representation extraction
        representations = self._collect_representations(X)
        representations_np = representations.numpy()

        # FIX #1: representations_np is now [R, N, F]; slice on dim 0.
        # Per-representation scores: shape (R, n_samples)
        score_partial = np.vstack([
            self.isolation_forest_list[r].new_decision_function(representations_np[r, :, :])
            for r in range(self.representations_number)
        ])  # shape: (R, n_samples)

        # Flow A — aggregation modes (#3)
        if aggregation == 'mean':
            final_scores = np.mean(score_partial, axis=0)

        elif aggregation == 'trimmed':
            trim_k = max(1, int(self.representations_number * 0.10))
            sorted_scores = np.sort(score_partial, axis=0)           # sort along R axis
            trimmed = sorted_scores[trim_k: self.representations_number - trim_k, :]
            final_scores = np.mean(trimmed, axis=0)

        elif aggregation == 'weighted':
            # Weight each representation by its variance across samples.
            # Higher-variance representations carry more discriminative signal.
            variances = np.var(score_partial, axis=1, keepdims=True)  # shape: (R, 1)
            total_var = variances.sum()
            if total_var == 0:
                # Degenerate case: all representations identical → fall back to mean
                final_scores = np.mean(score_partial, axis=0)
            else:
                weights = variances / total_var                        # normalised weights
                final_scores = np.sum(weights * score_partial, axis=0)

        else:
            raise ValueError(f"Unknown aggregation mode '{aggregation}'. "
                             f"Choose from: 'mean', 'trimmed', 'weighted'.")

        return final_scores

    def algorithm_name(self):
        if self.optimization == True:
            return "OptimizedDeepIF"
        else:
            return "DeepIF"
