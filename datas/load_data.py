import logging
from .load_cifar100 import load_partition_data_federated_cifar100
from .load_femnist import load_partition_data_federated_emnist
from .load_har import load_partition_data_federated_harbox
from .load_shakespeare import load_partition_data_federated_shakespeare
from .load_cinic import load_partition_data_cinic10



def load_dataloader(args):
    bz = args.batch_size
    dataset_name = args.dataset
    if dataset_name == 'cifar100':
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
        num_classes_dict
        ) = load_partition_data_federated_cifar100(args.data_cache_dir, batch_size=bz)
    elif dataset_name == 'femnist':
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
        num_classes_dict
        ) = load_partition_data_federated_emnist(args.data_cache_dir, batch_size=bz)
    elif dataset_name == 'shakespeare':
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
        num_classes_dict
        ) = load_partition_data_federated_shakespeare(args.data_cache_dir, batch_size=bz)
           
    elif dataset_name == 'cinic10':
        logging.info("load_data. dataset_name = %s" % dataset_name)  
        (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
        num_classes_dict
        ) = load_partition_data_cinic10(args.data_cache_dir, batch_size=bz)        
        
    elif dataset_name == 'harbox':
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
        client_num,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict, 
        class_num, 
        num_classes_dict  
        ) = load_partition_data_federated_harbox(args.data_cache_dir, batch_size=bz)
    else:
        raise ValueError("dataset: %s is not supported yet!" % dataset_name)
    
    dataset = [
        num_classes_dict,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict
    ]
    return dataset, class_num
