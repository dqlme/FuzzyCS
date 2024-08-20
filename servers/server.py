import copy
import random

import numpy as np
import torch
import wandb
import json
from selection import *
from clients.trainer import create_model_trainer
from clients.client import Client
from collections import defaultdict
import time
from datetime import datetime


current_time = datetime.now()
formatted_time = current_time.strftime("%Y%m%d-%H%M")

class FedAvgAPI(object):
    def __init__(self, args, device, dataset, model):
        self.recorder = defaultdict(lambda: {})
        self.recorder_file = "{}/{}_{}_{}.log".format(
            args.record_dir, args.record_file, args.client_sampling_method, formatted_time
        )
        self.recorder1 = defaultdict(lambda: {})
        self.recorder_time = "{}/{}_{}_{}_time.log".format(
            args.record_dir, args.record_file, args.client_sampling_method, formatted_time
        )        
        self.device = device
        self.args = args
        
        [   num_classes_dict,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict            
        ] = dataset
        self.val_global = None
        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.num_classes_dict = num_classes_dict
        # print("model = {}".format(model))
        self.model_trainer = create_model_trainer(model, args)
        self.model = model
        # print("self.model_trainer = {}".format(self.model_trainer))

        train_data_local_num = np.array(
            [int(i) for i in self.train_data_local_num_dict.values()]
        )
        # print(len(train_data_local_num))
        # set up computational heterogenity for all posible clients, the data size and
        # device capability has fixed correlation, it follows N(1, 0.25)
        # np.random.seed(self.args.random_seed)
        self.data_compute_correlation = np.random.normal(1.0, 0.25)
        self.compute_power = 100 + 10 *self.data_compute_correlation * (train_data_local_num - train_data_local_num.mean()) / (np.std(train_data_local_num)+1e-5)
        # self.compute_power = 1.0 + self.data_compute_correlation * 0.5 * train_data_local_num                           

        self.data_comm_correlation = np.random.normal(1.0, 0.25)
        self.comm_power = 100  + 10 * self.data_comm_correlation *  (train_data_local_num - train_data_local_num.mean()) / (np.std(train_data_local_num)+1e-5)
        # self.comm_power = 1.0 + self.data_comm_correlation * 0.5 * train_data_local_num

        # self.epoch_ddl = np.percentile(
        #     np.array(train_data_local_num) / self.compute_power,
        #     100 - self.args.stragglers_percent,
        # )
        num_classes_ = np.array(
            [int(i) for i in self.num_classes_dict.values()]
        )        
        self.recorder["clients"] = {
            "data": train_data_local_num.tolist(),
            "power": self.compute_power.tolist(),
            "num_class": num_classes_.tolist()
        }
        self.run_time = np.zeros(self.args.client_num_in_total)
        self._setup_clients(
            self.train_data_local_num_dict,
            self.train_data_local_dict,
            self.test_data_local_dict,
            self.compute_power,
            self.comm_power,            
            self.model_trainer,
            self.run_time
        )
        if self.args.client_sampling_method in ["DivFL", "DELTA"]:
            self.all_clients = []
            self._setup_all_clients(
            self.train_data_local_num_dict,
            self.train_data_local_dict,
            self.test_data_local_dict,
            self.compute_power,
            self.comm_power,
            self.model_trainer,
            self.run_time
        )
            
    def _setup_clients(
        self,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        compute_power,
        comm_power,
        model_trainer,
        run_time
    ):
        print("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            c = Client(
                client_idx,
                train_data_local_dict[client_idx],
                test_data_local_dict[client_idx],
                train_data_local_num_dict[client_idx],
                compute_power[client_idx],
                comm_power[client_idx],
                self.args,
                self.device,
                model_trainer,
                run_time[client_idx]
            )
            self.client_list.append(c)
        print("############setup_clients (END)#############")
    def _setup_all_clients(
        self,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        compute_power,
        comm_power,
        model_trainer,
        run_time
    ):
        # print("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_in_total):
            c = Client(
                client_idx,
                train_data_local_dict[client_idx],
                test_data_local_dict[client_idx],
                train_data_local_num_dict[client_idx],
                compute_power[client_idx],
                comm_power[client_idx],
                self.args,
                self.device,
                model_trainer,
                run_time[client_idx]
            )
            self.all_clients.append(c)
        # print("############setup_clients (END)#############")
    def train(self):
        # print("self.model_trainer = {}".format(self.model_trainer))
        self.w_global = self.model_trainer.get_model_params()
        
        for round_idx in range(self.args.comm_round + 1):
            time_every_round = time.time()
            # Test the initial loss
            if round_idx == 0:
                metric = self._local_test_on_all_clients(round_idx)
                continue

            print("################Communication round : {}".format(round_idx))

            w_locals = []
            
            client_indexes = self.client_sampling(
                round_idx, self.args.client_num_in_total, self.args.client_num_per_round, metric
            )
            self.recorder[round_idx]["selected_clients"] = client_indexes.tolist()
            self.recorder1[round_idx]['compute_power'] = self.compute_power.tolist()
            self.recorder1[round_idx]['commu_power'] = self.comm_power.tolist()
            # print("client_indexes = " + str(client_indexes))

            for idx, client in enumerate(self.client_list):
                # update dataset
                client_idx = client_indexes[idx]
                if random.random() < 0.5:
                    self.compute_power[client_idx] -= random.uniform(1, 10)
                    self.comm_power[client_idx] -= random.uniform(1, 10)
                else:
                    self.compute_power[client_idx] += random.uniform(1, 10)
                    self.comm_power[client_idx] += random.uniform(1, 10)

                # update the corresponding dataset for the selected client
                client.update_local_dataset(
                    client_idx,
                    self.train_data_local_dict[client_idx],
                    self.test_data_local_dict[client_idx],
                    self.train_data_local_num_dict[client_idx],
                    self.compute_power[client_idx],
                    self.comm_power[client_idx],
                    self.run_time[client_idx]
                )
                
                train_s = time.time()
                w = client.train(copy.deepcopy(self.w_global))
                # time.sleep(self.compute_power[client_idx] + self.comm_power[client_idx])
                running_time = time.time() - train_s 
                self.run_time[client_idx] = running_time
                if "train" not in self.recorder[round_idx]:
                    self.recorder[round_idx]["train"] = {}
                self.recorder[round_idx]["train"][idx] = running_time
                # print("Comm round {}: client {} training time: {}".format(round_idx, client_idx, running_time))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

            # update global weights
            if len(w_locals) > 0:
                self.w_global = self._aggregate(w_locals)
            self.model_trainer.set_model_params(self.w_global)

            # test results
            # at last round
            if round_idx == self.args.comm_round - 1:
                metric = self._local_test_on_all_clients(round_idx)
            # per {frequency_of_the_test} round
            elif round_idx % self.args.frequency_of_the_test == 0:
                metric = self._local_test_on_all_clients(round_idx)
            # print(len(metric))
            time_every_round = time.time()-time_every_round
            print(f'time consumption every round (s): {time_every_round}')



        with open(self.recorder_file, "w") as f:
            # print(self.recorder)
            f.write(json.dumps(dict(self.recorder)))
        
        with open(self.recorder_time, 'w') as f:
            f.write(json.dumps(dict(self.recorder1)))


    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round, metric=None, run_time=None):
        if self.args.client_sampling_method == "random":
            client_indexes = self.random_client_sampling(round_idx, client_num_in_total, client_num_per_round) 
        elif self.args.client_sampling_method == "AFL":
            AFL_s = ActiveFederatedLearning(self.args)
            client_indexes = AFL_s.select(round_idx, client_num_in_total, client_num_per_round, metric)
        elif self.args.client_sampling_method == "Powd":
            powd = Powd(self.args)
            powd.setup(self.train_data_local_num_dict)
            client_indexes = powd.select(round_idx, client_num_in_total, client_num_per_round)
        elif self.args.client_sampling_method == "DivFL":
            divfl = DivFL(self.args)
            model_copy = copy.deepcopy(self.model)
            model_copy.load_state_dict(self.w_global)
            client_indexes = divfl.select(round_idx, client_num_in_total, client_num_per_round, 
                                          model_copy,
                                          [c.model_trainer.model for c in self.all_clients])
        elif self.args.client_sampling_method == "MCDM":
            mcdm = MCDM(self.args)
            mcdm.setup(self.train_data_local_num_dict, self.num_classes_dict)
            # print(self.args.client_num_in_total)
            print(len(metric))
            # print(len(self.compute_power))
            # print(len(self.comm_power))
            criteria = np.array([metric, self.compute_power.tolist(), self.comm_power.tolist()])
            client_indexes = mcdm.select(round_idx, client_num_in_total, client_num_per_round, criteria)

        # elif self.args.client_sampling_method == "Oort":
        #     oort = Oort(self.args)
            
        elif self.args.client_sampling_method == "DELTA":
            client_grads = []
            for client in self.all_clients:
                gradients = client.train_num_epochs(copy.deepcopy(self.w_global), epoch_num=1)
                client_grads.append(gradients)
            delta = Delta(self.args)            
            client_indexes = delta.select(round_idx, client_num_in_total, client_num_per_round, client_grads, self.train_data_local_num_dict)
            # fedcor = FedCor(self.args)
        # elif self.args.client_sampling_method == "Pwo-d": 
        #     return self._iid_sampling(round_idx, client_num_in_total, client_num_per_round)
        else:
            raise NotImplementedError

        return client_indexes
    
    def random_client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = np.array(
                [client_index for client_index in range(client_num_in_total)]
            )
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(
                round_idx
            )  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(
                range(client_num_in_total), num_clients, replace=False
            )
        # print("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        test_data_num = len(self.test_global.dataset)
        sample_indices = random.sample(
            range(test_data_num), min(num_samples, test_data_num)
        )
        subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
        sample_testset = torch.utils.data.DataLoader(
            subset, batch_size=self.args.batch_size
        )
        self.val_global = sample_testset

    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def _aggregate_noniid_avg(self, w_locals):
        """
        The old aggregate method will impact the model performance when it comes to Non-IID setting
        Args:
            w_locals:
        Returns:
        """
        (_, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            temp_w = []
            for (_, local_w) in w_locals:
                temp_w.append(local_w[k])
            averaged_params[k] = sum(temp_w) / len(temp_w)
        return averaged_params

    def _local_test_on_all_clients(self, round_idx):

        print("################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        test_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        client = self.client_list[0]

        for client_idx in range(self.args.client_num_in_total):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(
                0,
                self.train_data_local_dict[client_idx],
                self.test_data_local_dict[client_idx],
                self.train_data_local_num_dict[client_idx],
                self.compute_power[client_idx],
                self.comm_power[client_idx],
                self.run_time[client_idx]
            )
            # train data
            # train_local_metrics = client.local_test(False)
            # train_metrics["num_samples"].append(
            #     copy.deepcopy(train_local_metrics["test_total"])
            # )
            # train_metrics["num_correct"].append(
            #     copy.deepcopy(train_local_metrics["test_correct"])
            # )
            # train_metrics["losses"].append(
            #     copy.deepcopy(train_local_metrics["test_loss"])
            # )

            # test data
            test_local_metrics = client.local_test(True)
            test_metrics["num_samples"].append(
                copy.deepcopy(test_local_metrics["test_total"])
            )
            test_metrics["num_correct"].append(
                copy.deepcopy(test_local_metrics["test_correct"])
            )
            test_metrics["losses"].append(
                copy.deepcopy(test_local_metrics["test_loss"])
            )

        # test on training dataset
        # train_acc = sum(train_metrics["num_correct"]) / sum(
        #     train_metrics["num_samples"]
        # )
        # train_loss = sum(train_metrics["losses"]) / sum(train_metrics["num_samples"])

        # test on test dataset
        ls = np.array(test_metrics["losses"])/np.array(test_metrics["num_samples"])
        # np.savetxt('ls.txt',ls,delimiter=',')
        test_acc = sum(test_metrics["num_correct"]) / sum(test_metrics["num_samples"])
        # print(len(np.array(test_metrics["num_correct"])))
        # print(len(np.array(test_metrics["num_samples"])))
        # print(np.array(test_metrics["num_samples"]))
        test_acc_std = np.std(np.array(test_metrics["num_correct"]) / np.array(test_metrics["num_samples"]))
        test_loss = sum(test_metrics["losses"]) / sum(test_metrics["num_samples"])
        

        # stats = {"training_acc": train_acc, "training_loss": train_loss}
        # self.recorder[round_idx] = {**self.recorder[round_idx], **stats}
        # if self.args.enable_wandb:
        #     wandb.log({"Train/Acc": train_acc, "round": round_idx})
        #     wandb.log({"Train/Loss": train_loss, "round": round_idx})

        # print(stats)

        stats = {"test_acc": test_acc, "test_loss": test_loss, "test_acc_std": test_acc_std}
        self.recorder[round_idx] = {**self.recorder[round_idx], **stats}
        if self.args.enable_wandb:
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})

        print(stats)
        return test_metrics["losses"]
