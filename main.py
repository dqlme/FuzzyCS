import torch
from models.cnn import DNN_HARBox, CNN_DropOut, MLP
from models.resnet import resnet18
from models.lstm import RNN_Shakespeare, LSTMModel, RNN_FedShakespeare
from models.mobilenet import mobilenet
import time
from datas.load_data import load_dataloader
from options import parse_args_and_yaml
from servers.server import FedAvgAPI


if __name__ == "__main__":
    args = parse_args_and_yaml()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(device)
    print(f'############## Method: {args.client_sampling_method} ##############')
    print(f'############## Dataset: {args.dataset} ##############')    
    dataloader,num_class = load_dataloader(args)
    
    if args.dataset == 'femnist':
        model = CNN_DropOut(False)
    elif args.dataset == 'harbox':
        model = DNN_HARBox()
    elif args.dataset == 'cifar100':
        model = resnet18(num_classes=num_class)    
    elif args.dataset == 'cinic10':
        model = resnet18(num_classes=num_class)            
    elif args.dataset == 'shakespeare':
        model = RNN_FedShakespeare()   
    else:
        raise NotImplementedError

    runner = FedAvgAPI(args, device, dataloader, model)
    st = time.time()
    runner.train()
    print("##################### Training Finished ######################")
    print("Time consumption (s):", time.time()-st)
