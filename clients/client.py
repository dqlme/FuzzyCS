import math
import torch
from torch import nn
import numpy as np


class Client:
    def __init__(
        self, 
        client_idx,
        local_training_data,
        local_test_data,
        local_sample_number,
        compute_power,
        comm_power,
        args,
        device,
        model_trainer,
        run_time=0.
    ):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.compute_power = compute_power
        self.comm_power = comm_power
        self.args = args
        self.device = device
        self.model_trainer = model_trainer
        self.run_time = run_time

    def update_local_dataset(
        self,
        client_idx,
        local_training_data,
        local_test_data,
        local_sample_number,
        compute_power,
        comm_power,
        run_time
    ):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.compute_power = compute_power
        self.comm_power = comm_power
        self.model_trainer.set_id(client_idx)
        self.run_time = run_time

    def get_sample_number(self):
        return self.local_sample_number

    def train(self, w_global):
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        return weights
    

    def train_init_epoch(self, w_global):
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train_init_epoch(
            self.local_training_data, self.device, self.args
        )
        weights = self.model_trainer.get_model_params()
        return weights

    def train_remaining_epochs(self, w_global):
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train_remaining_epochs(
            self.local_training_data, self.device, self.args
        )
        weights = self.model_trainer.get_model_params()
        return weights

    def train_num_epochs(self, w_global, epoch_num=None):
        self.model_trainer.set_model_params(w_global)
        gradients = self.model_trainer.train_num_epochs(self.local_training_data, self.device, self.args, epoch_num)
        # weights = self.model_trainer.get_model_params()
        return gradients

    def local_test(self, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics

    def coreset_grads(self, train_data):
        device = self.device
        args = self.args
        model = self.model_trainer.model

        gradients = []
        if args.model == "lr":
            for _, (x, target) in enumerate(train_data):
                gradients.append(x.flatten(1).cpu().numpy())
            return np.concatenate(gradients, axis=0)

        model.to(device)
        model.eval()
        criterion = nn.CrossEntropyLoss().to(device)

        for _, (x, target) in enumerate(train_data):
            x = x.to(device)
            target = target.to(device)
            with model.embedding_recorder:
                pred = model(x)
            loss = criterion(pred, target)  # pylint: disable=E1102

            with torch.no_grad():
                batch_num = target.shape[0]
                classes_num = pred.shape[1]
                embedding = model.embedding_recorder.embedding
                embedding_dim = model.get_last_layer().in_features
                bias_grads = torch.autograd.grad(loss, pred)[0]
                weights_grads = embedding.view(batch_num, 1, embedding_dim).repeat(
                    1, classes_num, 1
                )
                weights_grads *= bias_grads.view(batch_num, classes_num, 1).repeat(
                    1, 1, embedding_dim
                )
                tmp_grad = (
                    torch.cat([bias_grads, weights_grads.flatten(1)], dim=1)
                    .cpu()
                    .numpy()
                )
                gradients.append(tmp_grad)

        return np.concatenate(gradients, axis=0)

