from copy import deepcopy
import numpy as np
from .selection_base import SelectionBase
import torch
import random
import math

class Delta(SelectionBase):
    def __init__(self, args):
        self.beta = args.beta_ 

    def select(self, round_idx, client_num_in_total, client_num_per_round, client_grads, train_data_local_num_dict):
        if client_num_in_total == client_num_per_round:
            client_indexes = np.array(
                [client_index for client_index in range(client_num_in_total)]
            )
        else:        
            client_num_per_round = min(client_num_per_round, client_num_in_total)
            # np.random.seed(round_idx) # fix seed for reproducibility
            
            P = self.reweight_prac_theory(client_grads, train_data_local_num_dict)
            client_indexes = torch.multinomial(torch.tensor(P), client_num_per_round, False)
            client_indexes = client_indexes.numpy()
        return client_indexes.astype(int)

    def reweight_prac_theory(self, client_grads, train_data_local_num_dict):
        sum_num = 0
        Ls = []
        for num in train_data_local_num_dict.values():
            sum_num += num
            Ls.append(num)
        c = [0]*len(client_grads[0])
        for i in range(len(client_grads)):
            for j in range(len(c)):
                c[j] += client_grads[i][j]
        
        c = [i/len(client_grads) for i in c]
        for i in range(len(client_grads)):
            for j in range(len(c)):
                client_grads[i][j] = client_grads[i][j] - c[j]
        weights2 = [(self.beta * L / sum_num * sum([torch.norm(p) ** 2 for p in g]) ** 0.5 \
                      + (1-self.beta)*(L / sum_num)**0.5) for L, g in zip(Ls, client_grads)]
        sum_weights_2 = sum(weights2)
        P = [ w / sum_weights_2 for w in weights2]
        return P

