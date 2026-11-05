from copy import deepcopy
import numpy as np
from .selection_base import SelectionBase
# import torch

class FuzzyCS(SelectionBase):
    def setup(self, train_data_local_num_dict, num_classes_dict):
        client_ids = sorted(train_data_local_num_dict.keys())
        self.n_samples = np.array([train_data_local_num_dict[i] for i in client_ids])
        self.n_classes = np.array([num_classes_dict[i] for i in client_ids])
        
    def select(self, round_idx, client_num_in_total, client_num_per_round, metric):
        if client_num_in_total == client_num_per_round:
            client_indexes = np.array(
                [client_index for client_index in range(client_num_in_total)]
            )
        else:       
            mcdm_matrix = np.vstack((self.n_samples, self.n_classes, metric)).T
            normalized_mcmd = min_max_normalize(mcdm_matrix)
            client_num_per_round = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx) # fix seed for reproducibility
            
            # Use robust empirical percentiles for the requirement thresholds.
            # The first three criteria are benefit-oriented and the last two
            # criteria are cost-oriented, matching the manuscript definition.
            member_set = [
                np.quantile(normalized_mcmd[:, 0], 0.90),
                np.quantile(normalized_mcmd[:, 1], 0.90),
                np.quantile(normalized_mcmd[:, 2], 0.90),
                np.quantile(normalized_mcmd[:, 3], 0.10),
                np.quantile(normalized_mcmd[:, 4], 0.10),
            ]
            print(
                "FuzzyCS percentile thresholds (q90, q90, q90, q10, q10): "
                f"{member_set}"
            )
            probs = []
            for idx in range(client_num_in_total):
                lower_approx, upper_approx = get_appox(member_set, normalized_mcmd[idx])
                prob = 0.3 * lower_approx + 0.7 * upper_approx
                probs.append(prob)            
            # print(probs)
            probs = np.array(probs) / np.sum(probs)
            client_indexes = np.random.choice(client_num_in_total, client_num_per_round, p=probs, replace=False) 

        return client_indexes.astype(int)


def min_max_normalize(matrix):
    matrix_max = np.max(matrix, axis=0)
    matrix_min = np.min(matrix, axis=0)
    matrix_range = matrix_max - matrix_min
    matrix_range[matrix_range == 0] = 1.0
    matrix_norm = (matrix - matrix_min) / matrix_range
    return matrix_norm

def get_appox(M_set, criteria):
    '''
    M_set: Set of membership
    criteria: criteria matrix
    return: lower approx, upper approx
    '''
    assert len(M_set) == len(criteria)
    max_values, min_values = [], []
    for i in range(len(M_set)):
        max_v = np.max([M_set[i], 1-criteria[i]])
        min_v = np.min([M_set[i], criteria[i]])
        max_values.append(max_v)
        min_values.append(min_v)
    lower_approx = np.min(max_values)
    upper_approx = np.max(min_values)
    return lower_approx, upper_approx

