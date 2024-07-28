from copy import deepcopy
import numpy as np
from .selection_base import SelectionBase


class Powd(SelectionBase):
       
    def setup(self, train_data_local_num_dict):
        client_ids = sorted(train_data_local_num_dict.keys())
        n_samples = np.array([train_data_local_num_dict[i] for i in client_ids])
        self.weights = n_samples / np.sum(n_samples)
        
    def select(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = np.array(
                [client_index for client_index in range(client_num_in_total)]
            )
        else:        
            client_num_per_round = min(client_num_per_round, client_num_in_total)
            # np.random.seed(round_idx) # fix seed for reproducibility
            client_indexes = np.random.choice(client_num_in_total, client_num_per_round, p=self.weights, replace=False)          

        return client_indexes.astype(int)

