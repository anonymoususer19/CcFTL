from model import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch.utils.data as Data
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import random
import copy
from sklearn import metrics

def train_eval(flag, data_loader, model, optimizer, classifier_criterion):
    model.train() if flag == 'train' else model.eval()
    predicts = []
    groud_turth = []
    loss_num = 0.0
    for i, (x, y) in enumerate(data_loader):
        outputs = model(x)
        loss = classifier_criterion(outputs, y.squeeze())
        if flag == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        groud_turth.extend(y.detach().numpy().tolist())
        predicts.extend(np.argmax(outputs.detach().numpy(), axis=1).tolist())
        loss_num += loss.data
    loss_num = loss_num / len(data_loader)
    return groud_turth, predicts, loss_num

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="", help="data path")
    parser.add_argument("--DARKL_checkpoint_path", type=str, default="/checkpoint/", help="save DARKL module path")
    parser.add_argument("--UTP_checkpoint_path", type=str, default="/checkpoint/", help="save UTP module path")

    args = parser.parse_args()
    args.epochs = 50
    args.batch_size = 128
    args.lr = 0.01

    data = pd.read_csv(args.data_dir, encoding = 'utf-8')

    data_y = data['label']
    data_x = data.drop(['label'], axis=1)

    split_ratio = 0.8
    num = int(len(data_x) * split_ratio)
    data_x_train = data_x[:num]
    data_x_test = data_x[num:]
    data_y_train = data_y[:num]
    data_y_test = data_y[num:]

    target_x_train = torch.from_numpy(data_x_train.values).float()
    target_y_train = torch.from_numpy(data_y_train[:, np.newaxis])
    target_x_test = torch.from_numpy(data_x_test.values).float()
    target_y_test = torch.from_numpy(data_y_test[:, np.newaxis])

    torch.manual_seed(42)
    target_data_loader_train = Data.DataLoader(dataset=Data.TensorDataset(target_x_train, target_y_train),
                                               batch_size=args.batch_size, shuffle=True)
    target_data_loader_test = Data.DataLoader(dataset=Data.TensorDataset(target_x_test, target_y_test),
                                              batch_size=args.batch_size, shuffle=True)


    model_1 = torch.load(args.DARKL_checkpoint_path)
    model_2 = torch.load(args.UTP_checkpoint_path)
    model = CcFTL(data_x.shape[1], 5)

    pretrained_dict_1 = model_1.state_dict()
    pretrained_dict_2 = model_2.state_dict()
    model_dict = model.state_dict()

    pretrained_dict_1 = {k: v for k, v in pretrained_dict_1.items() if k in model_dict}
    pretrained_dict_2 = {k: v for k, v in pretrained_dict_2.items() if k in model_dict}

    model_dict.update(pretrained_dict_1)
    model_dict.update(pretrained_dict_2)
    model.load_state_dict(model_dict)

    for k, v in model.named_parameters():
        if (k == 'feature_extract_1.weight') or (k == 'feature_extract_1.bias') or (k == 'feature_extract_2.weight') or (k == 'feature_extract_2.bias') or (k == 'regression_1.weight') or (k == 'regression_1.bias') or (k == 'regression_2.weight') or (k == 'regression_2.bias') or (k == 'regression_predict.weight') or (k == 'regression_predict.bias'):
            v.requires_grad = False

    optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    criterion = nn.NLLLoss()

    max_p = 0.0
    max_r = 0.0
    max_f = 0.0
    min_epoch = 0
    for epoch in range(epochs):
        train_groud_turth, train_predicts, train_loss = train_eval('train', target_data_loader_train, model, optimizer,
                                                                   criterion)
        test_groud_turth, test_predicts, test_loss = train_eval('eval', target_data_loader_test, model, optimizer,
                                                                criterion)
        train_p = metrics.precision_score(train_groud_turth, train_predicts, average='macro')
        train_r = metrics.recall_score(train_groud_turth, train_predicts, average='macro')
        train_f = metrics.f1_score(train_groud_turth, train_predicts, average='macro')
        test_p = metrics.precision_score(test_groud_turth, test_predicts, average='macro')
        test_r = metrics.recall_score(test_groud_turth, test_predicts, average='macro')
        test_f = metrics.f1_score(test_groud_turth, test_predicts, average='macro')
        if test_f > max_f:
            max_f = test_f
            max_p = test_p
            max_r = test_r
            min_epoch = epoch
        print("epoch = %s, train: loss = %.4f, p = %.4f, r = %.4f, f = %.4f; test: loss = %.4f, p = %.4f, r = %.4f, f = %.4f" % (
            epoch, train_loss, train_p, train_r, train_f, test_loss, test_p, test_r, test_f))
    print("epoch = %s, max_p = %.4f, max_r = %.4f, max_f = %.4f" % (min_epoch, max_p, max_r, max_f))



