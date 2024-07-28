import torch
from torch import nn
import copy
from .model_trainer import ModelTrainerCLS

def create_model_trainer(model, args):
    model_trainer = CoresetModelTrainerCLS(model, args)
    return model_trainer


class CoresetModelTrainerCLS(ModelTrainerCLS):
    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        previous_model = copy.deepcopy(model.state_dict())
        # train and update
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        # criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []

            for _, data in enumerate(train_data):
                if len(data) == 3:
                    x, labels, weights = data
                    x, labels, weights = (
                        x.to(device),
                        labels.to(device),
                        weights.to(device),
                    )
                    model.zero_grad()
                    log_probs = model(x)
                    labels = labels.long()
                    loss = torch.div(
                        torch.inner(weights, criterion(log_probs, labels)),
                        torch.sum(weights),
                    )
                    fed_prox_reg = 0.0
                    if args.fedprox_mu != 0.0:
                        for name, param in model.named_parameters():
                            fed_prox_reg += (args.fedprox_mu / 2) * torch.norm(
                                (param - previous_model[name].data.to(device))
                            ) ** 2
                    loss += fed_prox_reg
                else:
                    x, labels = data
                    x, labels = x.to(device), labels.to(device)

                    model.zero_grad()
                    log_probs = model(x)
                    labels = labels.long()
                    loss = torch.mean(criterion(log_probs, labels))
                    fed_prox_reg = 0.0
                    if args.fedprox_mu != 0.0:
                        for name, param in model.named_parameters():
                            fed_prox_reg += (args.fedprox_mu / 2) * torch.norm(
                                (param - previous_model[name].data.to(device))
                            ) ** 2
                    loss += fed_prox_reg

                loss.backward()
                optimizer.step()

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # logging.info(
                #     "Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                #         epoch,
                #         (batch_idx + 1) * args.batch_size,
                #         len(train_data) * args.batch_size,
                #         100.0 * (batch_idx + 1) / len(train_data),
                #         loss.item(),
                #     )
                # )

                batch_loss.append(loss.item())
            if len(batch_loss) == 0:
                epoch_loss.append(0.0)
            else:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                

    def train_num_epochs(self, train_data, device, args, epochs_num=1):
        model = self.model

        model.to(device)
        model.train()

        previous_model = copy.deepcopy(model.state_dict())
        # train and update
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        gradients = [torch.zeros_like(p) for p in model.parameters()]
        # criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        epoch_loss = []
        # epochs_num = args.epochs if not epochs_num else epochs_num
        count = 0
        for epoch in range(epochs_num):
            batch_loss = []

            for _, data in enumerate(train_data):
                count += 1
                if len(data) == 3:
                    x, labels, weights = data
                    x, labels, weights = (
                        x.to(device),
                        labels.to(device),
                        weights.to(device),
                    )
                    model.zero_grad()
                    log_probs = model(x)
                    labels = labels.long()
                    loss = torch.div(
                        torch.inner(weights, criterion(log_probs, labels)),
                        torch.sum(weights),
                    )
                    fed_prox_reg = 0.0
                    if args.fedprox_mu != 0.0:
                        for name, param in model.named_parameters():
                            fed_prox_reg += (args.fedprox_mu / 2) * torch.norm(
                                (param - previous_model[name].data.to(device))
                            ) ** 2
                    loss += fed_prox_reg
                else:
                    x, labels = data
                    x, labels = x.to(device), labels.to(device)

                    model.zero_grad()
                    log_probs = model(x)
                    labels = labels.long()
                    loss = torch.mean(criterion(log_probs, labels))
                    fed_prox_reg = 0.0
                    if args.fedprox_mu != 0.0:
                        for name, param in model.named_parameters():
                            fed_prox_reg += (args.fedprox_mu / 2) * torch.norm(
                                (param - previous_model[name].data.to(device))
                            ) ** 2
                    loss += fed_prox_reg

                loss.backward()
                gradients = [g + p.grad.clone().detach() if p.grad is not None else g for g, p in zip(gradients, model.parameters())]
                optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) == 0:
                epoch_loss.append(0.0)
            else:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
        if count > 0:
            gradients = [g / count for g in gradients]
        return gradients


                   
                
                
                