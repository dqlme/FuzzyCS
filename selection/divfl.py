'''
Diverse Client Selection For Federated Learning via Submodular Maximization

Reference:
    https://openreview.net/pdf?id=nwKXyFvaUm
'''

from copy import deepcopy
import numpy as np
from .selection_base import SelectionBase
import numpy as np
import torch
from tqdm import tqdm
from itertools import product

class DivFL(SelectionBase):

    def select(self, round_idx, client_num_in_total, client_num_per_round, global_m, local_models):
        # metric: global_m, local_models

        if client_num_in_total == client_num_per_round:
            client_indexes = np.array(
                [client_index for client_index in range(client_num_in_total)]
            )
        else:        
            client_num_per_round = min(client_num_per_round, client_num_in_total)
            # np.random.seed(round_idx) # fix seed for reproducibility
            
            # get gradients
            local_grads = self.get_gradients(global_m, local_models)
            # get dissimilarity matrix, i.e., G(S)
            self.norm_diff = self.get_matrix_similarity_from_grads(local_grads)
            # stochastic greedy
            selected_clients = self.stochastic_greedy(client_num_in_total, client_num_per_round)
            client_indexes = np.array(selected_clients)
        return client_indexes

    def get_gradients(self, global_m, local_models):
        """
        return the `representative gradient` formed by the difference
        between the local work and the sent global model
        """
        local_model_params = []
        for model in local_models:
            local_model_params += [[tens.detach().cpu() for tens in list(model.parameters())]] #.numpy()

        global_model_params = [tens.detach().cpu() for tens in list(global_m.parameters())]

        local_model_grads = []
        for local_params in local_model_params:
            local_model_grads += [[local_weights - global_weights
                                   for local_weights, global_weights in
                                   zip(local_params, global_model_params)]]
        del local_model_params
        del global_model_params
        torch.cuda.empty_cache()
        return local_model_grads

    def get_matrix_similarity_from_grads(self, local_model_grads):
        """
        return the similarity matrix where the distance chosen to
        compare two clients is set with `distance_type`
        """
        n_clients = len(local_model_grads)
        metric_matrix = torch.zeros((n_clients, n_clients)).cpu()
        # for i, j in tqdm(product(range(n_clients), range(n_clients)), desc='>> similarity', total=n_clients**2, ncols=80):
        for i,j in product(range(n_clients), range(n_clients)):
            grad_1, grad_2 = local_model_grads[i], local_model_grads[j]
            for g_1, g_2 in zip(grad_1, grad_2):
                metric_matrix[i, j] += torch.sum(torch.square(g_1 - g_2))

        return metric_matrix

    def stochastic_greedy(self, num_total_clients, num_select_clients):
        # num_clients is the target number of selected clients each round,
        # subsample is a parameter for the stochastic greedy alg
        # initialize the ground set and the selected set
        V_set = set(range(num_total_clients))
        SUi = set()

        for ni in range(num_select_clients):
            if num_select_clients < len(V_set):
                R_set = np.random.choice(list(V_set), num_select_clients, replace=False)  # S
            else:
                R_set = list(V_set)
            if ni == 0:
                marg_util = self.norm_diff[:, R_set].sum(0)  
                i = marg_util.argmin() # min||fk-fi||(i \in S)
                client_min = self.norm_diff[:, R_set[i]]    # client_min: G(S)
            else:
                client_min_R = torch.minimum(client_min[:, None], self.norm_diff[:, R_set]) #client_min_R: G(S\cup{k})
                marg_util = client_min_R.sum(0)
                i = marg_util.argmin()
                client_min = client_min_R[:, i]
            SUi.add(R_set[i])
            V_set.remove(R_set[i])
        return list(SUi)
