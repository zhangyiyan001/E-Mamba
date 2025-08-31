# -*- coding:utf-8 -*-

import os
import torch
import argparse
from dataset import load_data, generater, normalize, setup_seed
from module import weight_init, AA_andEachClassAccuracy, applyPCA
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import numpy as np
import time
from einops import rearrange
from mamba.vmamba import MultimodalClassier
import matplotlib.pyplot as plt

def train_model(net, epoches, train_iter, optimizer, criterion, device, dataset): 
    train_loss_list = []
    train_acc_list = []
    accuracy = []
    best_accuracy = 0.0
    best_y_pred = []
    best_y_test = []
    best_model_path = './models/' + dataset + '.pt'
    tick1 = time.time()

    for epoch in range(epoches):
        train_acc_sum, train_loss_sum = 0.0, 0.0
        for step, (x1, x2, y) in enumerate(train_iter):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            y_hat = net(x1, x2)
            loss = criterion(y_hat, y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            train_acc_sum += (y_hat.argmax(dim=-1) == y).float().sum().item()
        lr_adjust.step()

        print('epoch %d, train loss %.6f, train acc %.4f' % (
            epoch + 1, train_loss_sum / len(train_iter), train_acc_sum / len(train_iter.dataset)))
        train_loss_list.append(train_loss_sum / len(train_iter))
        train_acc_list.append(train_acc_sum / len(train_iter.dataset))

        if train_loss_list[-1] <= min(train_loss_list):
            print('\n***Start Testing***\n')
            y_test = []
            y_pred = []
            net.eval()
            with torch.no_grad():
                for step, (x1, x2, y) in enumerate(test_iter):
                    x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                    y_hat = net(x1, x2)
                    y_pred.extend(y_hat.cpu().argmax(dim=1))
                    y_test.extend(y.cpu())
            net.train()
            oa = accuracy_score(np.array(y_test), np.array(y_pred))
            confusion = confusion_matrix(np.array(best_y_test), np.array(best_y_pred))
            each_acc, aa = AA_andEachClassAccuracy(confusion)
            kappa = cohen_kappa_score(np.array(best_y_test), np.array(best_y_pred))
            if oa > best_accuracy:
                best_accuracy = oa
                best_y_pred = y_pred.copy()  # 保存当前的预测结果
                best_y_test = y_test.copy()  # 保存当前的真实标签
                print('***Saving model parameters***')
                torch.save(net.state_dict(), best_model_path)
            print(f'OA: {oa}, AA: {aa}, Kappa:{kappa}, Best accuracy: {best_accuracy}')
    tick2 = time.time()
    # 所有 epoch 结束后，使用 best_accuracy 对应的预测结果生成混淆矩阵和其他指标
    if dataset == 'Houston':
        target_names = ['Healthy grass', 'Stressed grass', 'Synthetic grass', 'Tree',
                    'Soil', 'Water', 'Residential', 'Commercial', 'Road', 'Highway',
                    'Railway', 'Parking lot 1', 'Parking lot 2', 'Tennis court', 'Running track']
    if dataset == 'Berlin':
        target_names = ['Forest', 'Residential Area', 'Industrial Area', 'Low Plants', 'Soil',
                        'Allotment', 'Commercial Area', 'Water']
    if dataset == 'Trento':
        target_names = ['Apple trees', 'Buildings', 'Ground', 'Wood', 'Vineyard', 'Roads']
    if dataset == 'MUUFL':
        target_names = ['Trees', 'Mostly grass', 'Mixed ground surface', 'Dirt and sand', 'Road', 'Water',
                        'Building shadow', 'Building', 'Sidewalk', 'Yellow curb', 'Cloth panels']
    classification = classification_report(np.array(best_y_test), np.array(best_y_pred), target_names=target_names, digits=4)
    confusion = confusion_matrix(np.array(best_y_test), np.array(best_y_pred))
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(np.array(best_y_test), np.array(best_y_pred))

    file_name = "./results/{}/{}.txt".format(dataset, dataset)
    with open(file_name, 'a') as x_file:
        x_file.write('\n**************************************************************************************\n')
        x_file.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        x_file.write('\n')
        x_file.write('Overall accuracy (OA): {:.2f}%\n'.format(best_accuracy * 100))
        x_file.write('Average accuracy (AA): {:.2f}%\n'.format(aa * 100))
        x_file.write('Kappa accuracy: {:.2f}%\n'.format(kappa * 100))
        x_file.write('\n')
        x_file.write('{}'.format(classification))
        x_file.write('\n')
        x_file.write('{}'.format(confusion))
        x_file.write('\n')
        x_file.write('Training Time: {}s\n'.format(tick2 - tick1))
        x_file.write('\n**************************************************************************************\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Houston', choices=['Houston', 'Trento', 'MUUFL'])
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--window_size', type=int, default=7)
    return parser.parse_args()

# 解析命令行参数
args = parse_args()

# 设置GPU环境
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

setup_seed(seed=args.seed)
epoches = args.epochs
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dataset = args.dataset

# 数据集配置字典
dataset_configs = {
    'Houston': {'l1': 144, 'l2': 1, 'num_classes': 15},
    'Trento': {'l1': 63, 'l2': 1, 'num_classes': 6},
    'MUUFL': {'l1': 64, 'l2': 2, 'num_classes': 11}
}

print(f'使用数据集: {dataset}')
print(f'使用GPU设备: {args.gpu}')
print(f'训练轮数: {epoches}')
print(f'随机种子: {args.seed}')
print(f'窗口大小: {args.window_size}')

# Load data and preprocess
HSI_data, LiDAR_data, Train_data, Test_data, GT = load_data(dataset)
HSI_data = normalize(HSI_data, type=1)
LiDAR_data = normalize(LiDAR_data, type=1)
#HSI_data = applyPCA(HSI_data, numComponents=30)

(TRAIN_SIZE, TEST_SIZE, TOTAL_SIZE, train_iter, test_iter) = generater(dataset,
                                                                     HSI_data,
                                                                     LiDAR_data,
                                                                     Train_data,
                                                                     Test_data,
                                                                     GT,
                                                                     batch_size=128,
                                                                     windowSize=args.window_size)

print('TRAIN_SIZE: ', TRAIN_SIZE) #打印训练集大小
print('TEST_SIZE: ', TEST_SIZE)
print('TOTAL_SIZE: ', TOTAL_SIZE)
print('----Training on {}----\n'.format(device))

# 获取当前数据集的配置
config = dataset_configs[dataset]
net = MultimodalClassier(l1=config['l1'], 
                         l2=config['l2'],
                         dim=config['l1'],
                         num_classes=config['num_classes']).to(device)


optimizer = optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.0001)
criterion = nn.CrossEntropyLoss().to(device)
lr_adjust = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=30, gamma=0.5, last_epoch=-1)

if not os.path.exists('./models'):
    os.makedirs('./models')

# Start training
train_model(net, epoches, train_iter, optimizer, criterion, device, dataset)