# coding:utf-8

import numpy as np
import pandas as pd
import time
import json
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from model import *

from client import local_train_1, local_train_2, local_test_1, local_test_2
from server import global_communicate

def main(net1, net2):
    min_mae = 0.0
    min_mse = 100000000.0
    max_p = 0.0
    max_r = 0.0
    max_f = 0.0
    chkpt1 = {
        "step": 0,
        "epochs": args.epochs,
        "min_mae": min_mae,
        "min_mse": min_mse,
        "model": net1.module.state_dict() if type(net1) is nn.parallel.DistributedDataParallel else net1.state_dict(),
        'optimizer': optimizer1.state_dict()
    }
    chkpt2 = {
        "step": 0,
        "epochs": args.epochs,
        "max_p": max_p,
        "max_r": max_r,
        "max_f": max_f,
        "model": net2.module.state_dict() if type(net2) is nn.parallel.DistributedDataParallel else net2.state_dict(),
        'optimizer': optimizer2.state_dict()
    }

    for step in range(1, args.steps + 1):
        if not args.is_master:
            net1 = local_train_1(net1, optimizer1, regression_criterion, classifier_criterion, miss_data_loader_train, step, args)
            net2 = local_train_2(net2, optimizer2, classifier_criterion, data_loader_train, step, args)

        model1, model2 = global_communicate(net1, net2, dist, args)

        if not args.is_master:
            test_mae, test_mse = local_test_1(model1, regression_criterion, classifier_criterion, miss_data_loader_test, args)
            test_p, test_r, test_f = local_test_2(model2, classifier_criterion, data_loader_test, args)

            if test_mse < min_mse:
                min_mae = test_mae
                min_mse = test_mse
                chkpt1 = {"step": step, "epochs": args.epochs, "min_mse": min_mse, "min_mae": min_mae,"model": model1.module.state_dict() if type(model1) is nn.parallel.DistributedDataParallel else model1.state_dict(),'optimizer': optimizer1.state_dict()}

            if test_f > max_f:
                max_p = test_p
                max_r = test_r
                max_f = test_f
                chkpt2 = {"step": step, "epochs": args.epochs, "max_p": max_p, "max_r": max_r, "max_f": max_f, "model": model2.module.state_dict() if type(model2) is nn.parallel.DistributedDataParallel else model2.state_dict(),'optimizer': optimizer2.state_dict()}

    if not os.path.exists(args.DARKL_checkpoint_path):
        os.makedirs(args.DARKL_checkpoint_path)
    if not os.path.exists(args.UTP_checkpoint_path):
        os.makedirs(args.UTP_checkpoint_path)
    torch.save(chkpt1, os.path.join(args.DARKL_checkpoint_path + "best_DARKL_module_{}.pt".format(str_time)))
    torch.save(chkpt2, os.path.join(args.UTP_checkpoint_path + "best_UTP_module_{}.pt".format(str_time)))

if __name__ == "__main__":
    current_time = time.asctime(time.localtime(time.time()))
    str_time = '_'.join(current_time.split(' '))

    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type = int, default = 0, help = "rank number")
    parser.add_argument("--data_dir", type = str, default = "", help = "data path")
    parser.add_argument("--DARKL_checkpoint_path", type = str, default = "/checkpoint/", help = "save DARKL module path")
    parser.add_argument("--UTP_checkpoint_path", type=str, default="/checkpoint/", help="save UTP module path")

    args = parser.parse_args()
    args.steps = 200
    args.epochs = 1
    args.batch_size = 128
    args.lr = 0.01
    args.world_size = 3
    args.device = 'cpu'

    init_method = "tcp://0.0.0.0:22248"
    backend = "gloo"
    world_size = 3
    dist.init_process_group(init_method = init_method, backend = backend, world_size = 3, rank = args.rank)
    args.master_rank = 0
    args.is_master = (dist.get_rank() == args.master_rank)
    
    print("Current rank: ", dist.get_rank())
    print("Is master: ", dist.get_rank() == args.master_rank)
    print("All size: ", dist.get_world_size())

    if not args.is_master:
        data = pd.read_csv(args.data_dir, encoding = 'utf-8')
        data = data.rename(columns={'jd_pin_num':'miss_label'})
        data['domain_label'] = dist.get_rank() - 1

        data_y = data['label']
        data_x = data.drop(['label', 'domain_label'], axis=1)

        data_miss_y = data['miss_label']
        data_domain_y = data['domain_label']
        data_miss_x = data.drop(['miss_label', 'domain_label', 'label'], axis = 1)

        split_ratio = 0.8
        num = int(len(data_x) * split_ratio)
        data_x_train = data_x[:num]
        data_x_test = data_x[num:]
        data_y_train = data_y[:num]
        data_y_test = data_y[num:]

        data_miss_x_train = data_miss_x[:num]
        data_miss_x_test = data_miss_x[num:]
        data_miss_y_train = data_miss_y[:num]
        data_miss_y_test = data_miss_y[num:]
        data_domain_y_train = data_domain_y[:num]
        data_domain_y_test = data_domain_y[num:]
        
        x_train = torch.from_numpy(data_x_train.values).float()
        x_test = torch.from_numpy(data_x_test.values).float()
        y_train = torch.from_numpy(data_y_train[:, np.newaxis])
        y_test = torch.from_numpy(data_y_test[:, np.newaxis])

        miss_x_train = torch.from_numpy(data_miss_x_train.values).float()
        miss_x_test = torch.from_numpy(data_miss_x_test.values).float()
        miss_y_train = torch.from_numpy(data_miss_y_train[:, np.newaxis]).float()
        miss_y_test = torch.from_numpy(data_miss_y_test[:, np.newaxis]).float()
        domain_y_train = torch.from_numpy(data_domain_y_train[:, np.newaxis])
        domain_y_test = torch.from_numpy(data_domain_y_test[:, np.newaxis])

        torch.manual_seed(42)
        data_loader_train = Data.DataLoader(dataset = Data.TensorDataset(x_train, y_train), batch_size = args.batch_size, shuffle = True)
        data_loader_test = Data.DataLoader(dataset = Data.TensorDataset(x_test, y_test), batch_size = args.batch_size, shuffle = True)

        miss_data_loader_train = Data.DataLoader(dataset = Data.TensorDataset(miss_x_train, miss_y_train, domain_y_train), batch_size = args.batch_size, shuffle = True)
        miss_data_loader_test = Data.DataLoader(dataset = Data.TensorDataset(miss_x_test, miss_y_test, domain_y_test), batch_size = args.batch_size, shuffle = True)

    DARKL = first_Model(45, 256, 128, 64, 32, 64, 1, args.world_size - 1).to(args.device)
    UTP = ClassifierModel(46, 128, 64, 5).to(args.device)

    optimizer1 = optim.SGD(DARKL.parameters(), lr = args.lr)
    optimizer2 = optim.SGD(UTP.parameters(), lr = args.lr)
    regression_criterion = nn.MSELoss()
    classifier_criterion = nn.NLLLoss()

    main(DARKL, UTP)
