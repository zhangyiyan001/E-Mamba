# -*- coding:utf-8 -*-

import torch
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import torch.utils.data as Data
from module import get_image_cubes
from dataset import load_data, normalize
from net import Network
from mamba.vmamba import MultimodalClassier

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def generate_dataset(X_hsi, X_lidar, GT, batch_size, windowSize):
    X_all_hsi, Y_all_hsi = get_image_cubes(X_hsi, GT, windowSize)
    X_all_lidar, Y_all_lidar = get_image_cubes(X_lidar, GT, windowSize)

    X_all_hsi = X_all_hsi.astype(np.float32)
    X_all_lidar = X_all_lidar.astype(np.float32)

    hsi_all = torch.from_numpy(X_all_hsi).type(torch.FloatTensor)
    lidar_all = torch.from_numpy(X_all_lidar).type(torch.FloatTensor)
    y_all = torch.from_numpy(Y_all_hsi).type(torch.int64)

    torch_all = Data.TensorDataset(hsi_all, lidar_all, y_all)
    TOTAL_SIZE = y_all.shape[0]
    print(f'Total_size:{TOTAL_SIZE}')
    all_iter = Data.DataLoader(
        dataset=torch_all,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return TOTAL_SIZE, all_iter
def colormap(num_class, p):
    if p == True:
        # cdict = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF', '#C0C0C0',
        #          '#808080', '#800000', '#808000', '#008000', '#800080', '#008080', '#000080', '#FFA500', '#FFD700']
        cdict = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#C86400', '#00C864',
                 '#6400C8', '#C80064', '#64C800', '#0064C8', '#964B4B', '#4B964B', '#4B4B96', '#FF6464']
        return colors.ListedColormap(cdict, N=num_class)
    else:
        # cdict = ['#000000', '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF', '#C0C0C0', '#808080',
        #         '#800000', '#808000', '#008000', '#800080', '#008080', '#000080', '#FFA500', '#FFD700']
        cdict = ['#000000', '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#C86400', '#00C864',
                 '#6400C8', '#C80064', '#64C800', '#0064C8', '#964B4B', '#4B964B', '#4B4B96', '#FF6464']
        return colors.ListedColormap(cdict, N=num_class+1)

def dis_groundtruth(dataset, num_class, gt, p):
    '''plt.figure(title)
    plt.title(title)'''
    plt.imshow(gt, cmap=colormap(num_class, p=p))
    # spectral.imshow(classes=gt)
    '''plt.colorbar()'''
    plt.xticks([])
    plt.yticks([])
    '''plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)'''
    if p:
        plt.savefig('./results/{}/{}.png'.format(dataset, dataset+'true'), dpi=1200, pad_inches=0.0)
    else:
        plt.savefig('./results/{}/{}.png'.format(dataset, dataset+'false'), dpi=1200, pad_inches=0.0)
    plt.show()

def plot(dataset, TOTAL_SIZE, all_iter, Y, model, device):
    all_pixels = np.where(Y != 0)  # 只选择非背景像素（标签不为0）
    all_category = np.zeros([TOTAL_SIZE, 1])
    result_pic = np.zeros_like(Y)
    all_pred = []
    num_class = np.unique(Y).shape[0] - 1
    for step, (X1, X2, y) in enumerate(all_iter):
        model.eval()
        model.to(device)
        X1 = X1.to(device)
        X2 = X2.to(device)
        pred = model(X1, X2)
        all_pred.extend(pred.cpu().argmax(axis=1))
    all_category[:, 0] = all_pred[:]
    for k in range(TOTAL_SIZE):
        row = all_pixels[0][k]  # 得到像素点标签的横坐标
        col = all_pixels[1][k]
        result_pic[row, col] = all_category[k, 0] + 1
    print(np.unique(result_pic))
    dis_groundtruth(dataset=dataset, num_class=num_class, gt=result_pic, p=False)

if __name__ == '__main__':
    dataset = 'MUUFL'
    model_path = './models/' + dataset + '.pt'
    net = MultimodalClassier(l1=64, l2=2, dim=64, num_classes=11)
    net.load_state_dict(torch.load(model_path))
    HSI_data, LiDAR_data, Train_data, Test_data, GT = load_data(dataset)
    HSI_data = normalize(HSI_data, type=1)
    LiDAR_data = normalize(LiDAR_data, type=1)
    TOTAL_SIZE, all_iter = generate_dataset(HSI_data, LiDAR_data, GT, 128, 7)
    plot(dataset, TOTAL_SIZE, all_iter, GT, net, device)
