from copy import deepcopy
import numpy as np
from .selection_base import SelectionBase
# import torch

class MCDM(SelectionBase):
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
            normalized_mcmd = normalize(mcdm_matrix)
            client_num_per_round = min(client_num_per_round, client_num_in_total)
            # np.random.seed(round_idx) # fix seed for reproducibility
            
            memeber_set = [np.max(normalized_mcmd[0]),np.max(normalized_mcmd[1]),np.max(normalized_mcmd[2]),\
                           np.min(normalized_mcmd[3]),np.min(normalized_mcmd[4])]
            probs = []
            for idx in range(client_num_in_total):
                lower_approx, upper_approx = get_appox(memeber_set, normalized_mcmd[idx])
                prob = self.args.alpha_ * lower_approx + (1-self.args.alpha_) * upper_approx
                probs.append(prob)            
            # print(probs)
            probs = np.array(probs) / np.sum(probs)
            client_indexes = np.random.choice(client_num_in_total, client_num_per_round, p=probs, replace=False) 

        return client_indexes.astype(int)


def min_max_normalize(matrix):
    matrix_max = np.max(matrix, axis=0)
    matrix_min = np.min(matrix, axis=0)
    matrix_norm = (matrix - matrix_min) / (matrix_max - matrix_min)
    return matrix_norm

def normalize(matrix):
    m, n = matrix.shape
    matrix_norm = np.zeros((m,n))
    for i in range(n):
        matrix_norm[:,i] =np.power(matrix[:,i],2) / np.sum(np.power(matrix[:,i],2))
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

