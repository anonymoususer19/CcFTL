# coding: utf-8
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn import metrics

def local_train_1(net, optimizer, regression_criterion, classifier_criterion, train_dataloader, step, args):
    if not args.is_master:
        net.train()
        for epoch in range(1, args.epochs + 1):
            train_predicts = []
            train_groud_turth = []
            loss_num = 0.0
            for batch, (data, target_1, target_2) in enumerate(train_dataloader):
                outputs_1, outputs_2 = net(data, 0.6)
                loss = regression_criterion(outputs_1, target_1) + classifier_criterion(outputs_2, target_2.squeeze())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_groud_turth.extend(target_1.detach().numpy().tolist())
                train_predicts.extend(outputs_1.detach().numpy().tolist())
                loss_num += loss.data

            loss_num = loss_num / len(train_dataloader)
            train_mae = mean_absolute_error(np.array(train_groud_turth), np.array(train_predicts))
            train_mse = mean_squared_error(np.array(train_groud_turth), np.array(train_predicts))
            
            print("step = %s, epoch = %s, train: loss = %.4f, mae = %.4f, mse = %.4f"% (step, epoch, loss_num, train_mae, train_mse))

    return net

def local_test_1(model, regression_criterion, classifier_criterion, test_dataloader, args):
    with torch.no_grad():
        model.eval()
        test_predicts = []
        test_groud_turth = []
        loss_num = 0.0
        for _, (data, target_1, target_2) in enumerate(test_dataloader):
            outputs_1, outputs_2 = model(data, 0.6)
            
            loss = regression_criterion(outputs_1, target_1) + classifier_criterion(outputs_2, target_2.squeeze())

            test_groud_turth.extend(target_1.detach().numpy().tolist())
            test_predicts.extend(outputs_1.detach().numpy().tolist())
            loss_num += loss.data

        loss_num = loss_num / len(test_dataloader)
        test_mae = mean_absolute_error(np.array(test_groud_turth), np.array(test_predicts))
        test_mse = mean_squared_error(np.array(test_groud_turth), np.array(test_predicts))
        
        print("test: loss = %.4f, mae = %.4f, mse = %.4f"% (loss_num, test_mae, test_mse))
        
    return test_mae, test_mse


def local_train_2(net, optimizer, classifier_criterion, train_dataloader, step, args):
    if not args.is_master:
        net.train()
        for epoch in range(1, args.epochs + 1):
            train_predicts = []
            train_groud_turth = []
            loss_num = 0.0
            for batch, (data, target) in enumerate(train_dataloader):
                outputs = net(data)
                loss = classifier_criterion(outputs, target.squeeze())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_groud_turth.extend(target.detach().numpy().tolist())
                train_predicts.extend(np.argmax(outputs.detach().numpy(), axis = 1).tolist())
                loss_num += loss.data

            loss_num = loss_num / len(train_dataloader)
            train_p = metrics.precision_score(train_groud_turth, train_predicts, average='macro')
            train_r = metrics.recall_score(train_groud_turth, train_predicts, average='macro')
            train_f = metrics.f1_score(train_groud_turth, train_predicts, average='macro')

            print("step = %s, epoch = %s, train: loss = %.4f, p = %.4f, r = %.4f, f = %.4f" % (
            step, epoch, loss_num, train_p, train_r, train_f))

    return net


def local_test_1(model, classifier_criterion, test_dataloader, args):
    with torch.no_grad():
        model.eval()
        test_predicts = []
        test_groud_turth = []
        loss_num = 0.0
        for _, (data, target) in enumerate(test_dataloader):
            outputs = model(data)

            loss = classifier_criterion(outputs, target.squeeze())

            test_groud_turth.extend(target.detach().numpy().tolist())
            test_predicts.extend(np.argmax(outputs.detach().numpy(), axis = 1).tolist())
            loss_num += loss.data

        loss_num = loss_num / len(test_dataloader)
        test_p = metrics.precision_score(test_groud_turth, test_predicts, average='macro')
        test_r = metrics.recall_score(test_groud_turth, test_predicts, average='macro')
        test_f = metrics.f1_score(test_groud_turth, test_predicts, average='macro')

        print("test: loss = %.4f, p = %.4f, r = %.4f, f = %.4f" % (loss_num, test_p, test_r, test_f))

    return test_p, test_r, test_f

if __name__ == "__main__":
    pass



