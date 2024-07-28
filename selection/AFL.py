from copy import deepcopy
import numpy as np
from .selection_base import SelectionBase


class ActiveFederatedLearning(SelectionBase):
    def __init__(self, args):
        self.alpha1 = args.alpha1 #0.75
        self.alpha2 = args.alpha2 #0.01
        self.alpha3 = args.alpha3 #0.1

    def select(self, round_idx, client_num_in_total, client_num_per_round, metric):
        if client_num_in_total == client_num_per_round:
            client_indexes = np.array(
                [client_index for client_index in range(client_num_in_total)]
            )
        else:        
            client_num_per_round = min(client_num_per_round, client_num_in_total)
            # np.random.seed(round_idx) # fix seed for reproducibility
            
            # set sampling distribution
            values = np.exp(np.array(metric) * self.alpha2)
            #  select 75% of K(total) users
            num_drop = client_num_in_total - int(self.alpha1 * client_num_in_total)
            drop_client_idxs = np.argsort(metric)[:num_drop]
            probs = deepcopy(values)
            probs[drop_client_idxs] = 0
            probs /= sum(probs)
            #probs = np.nan_to_num(probs, nan=max(probs))
            #  select 99% of m users using prob.
            num_select = int((1 - self.alpha3) * client_num_per_round)
            selected = np.random.choice(len(metric), num_select, p=probs, replace=False)
            # select users randomly
            not_selected = np.array(list(set(np.arange(client_num_in_total)) - set(selected)))
            selected2 = np.random.choice(not_selected, client_num_per_round - num_select, replace=False)
            client_indexes = np.append(selected, selected2, axis=0)

        return client_indexes.astype(int)

