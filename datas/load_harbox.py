import warnings
import os
import os.path
import numpy as np
import math
from collections import OrderedDict
import random
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import logging
import torch

class HAR():
    class_set = ['Call','Hop','typing','Walk','Wave']
    label = [0,1,2,3,4] 
    DIMENSION_OF_FEATURE = 900
    NUM_OF_TOTAL_USERS = 120
    count_user_data = np.zeros(NUM_OF_TOTAL_USERS)
    classes = []

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, train=True, transform=None, target_transform=None, imgview=False,num_classes=5):
        
        self.train = train  # training set or test set
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train_test_ratio=0.9
        self.client_mapping=OrderedDict()
        self.client_label_distribution=OrderedDict()
        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You have to download it')

        # load class information
        self.data, self.targets = self.load_data(num_classes)


        # load data and targets

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def processed_folder(self):
        return self.root

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(self.processed_folder))

    def load_data(self,num_classes):

        # dataset append and split

        coll_class = []
        coll_label = []

        total_class = 0
        for current_user_id in range(1,self.NUM_OF_TOTAL_USERS+1):
            current_path = os.path.join(self.processed_folder, str(current_user_id))
            total_class_per_user=0
            cur_label_distribution=[0]*num_classes
            for class_id in range(num_classes):
                current_file = os.path.join(current_path, str(self.class_set[class_id]) + '_train.txt')

                if os.path.exists(current_file):

                    temp_original_data = np.loadtxt(current_file)
                    temp_reshape = temp_original_data.reshape(-1, 100, 10)
                    temp_coll = temp_reshape[:, :, 1:10].reshape(-1, self.DIMENSION_OF_FEATURE)
                    random.shuffle(temp_coll)
                    count_img = math.floor(temp_coll.shape[0]*self.train_test_ratio)
                    # print(temp_original_data.shape)
                    # print(temp_coll.shape)
                    cur_label_distribution[class_id]=count_img
                    if self.train:                       
                        temp_label = class_id * np.ones(count_img, dtype=int)
                        coll_class.extend(temp_coll[:count_img,:])
                        coll_label.extend(temp_label)
                        total_class_per_user+=count_img
                    else:
                        temp_label = class_id * np.ones(temp_coll.shape[0]-count_img-1, dtype=int)
                        coll_class.extend(temp_coll[count_img+1:,:])
                        coll_label.extend(temp_label)
                        total_class_per_user+=temp_coll.shape[0]-count_img-1
                    
            self.client_label_distribution[current_user_id-1]=cur_label_distribution
            self.client_mapping[current_user_id-1]=[i for i in range(total_class,total_class+total_class_per_user)]
            total_class+=total_class_per_user

        coll_class = np.array(coll_class)
        coll_label = np.array(coll_label)

        return coll_class, coll_label
    
    
DEFAULT_BATCH_SIZE = 32
DEFAULT_TRAIN_CLIENTS_NUM = 121

def load_partition_data_federated_harbox(
    data_dir, batch_size=DEFAULT_BATCH_SIZE
):
    # data_dir = './dataset/large_scale_HARBox/'
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    # test_data_local_dict = dict()
    train_dataset = HAR(root=data_dir, train=True, transform=None,num_classes=5)
    test_dataset = HAR(root=data_dir, train=False, transform=None,num_classes=5)
    indices = torch.load('harbox_121.pt')
    
    # trainloader = DataLoader(train_dataset, batch_size=DEFAULT_BATCH_SIZE, shuffle=True, drop_last=True)
    testloader = DataLoader(test_dataset,batch_size=batch_size, shuffle=False, drop_last=False)
    for client_idx in range(DEFAULT_TRAIN_CLIENTS_NUM):
        trainset_local = Subset(train_dataset, indices[client_idx])
        train_data_local = DataLoader(trainset_local, batch_size=batch_size, shuffle=True, drop_last=True)
        local_data_num = len(train_data_local.dataset)
        data_local_num_dict[client_idx] = local_data_num
        logging.info(
            "client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num)
        )
        train_data_local_dict[client_idx] = train_data_local
    return (
        DEFAULT_TRAIN_CLIENTS_NUM,
        len(train_dataset),
        len(test_dataset),
        testloader,
        data_local_num_dict,
        train_data_local_dict
    )   