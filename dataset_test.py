import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset


def minmax(x):
    if x.name=='cnt':
        # return (x - cnt_min) / (cnt_max - cnt_min)
        # return x
        return np.log(x)
    # if x.name=='yr':
    #     return x
    else:
        return np.log(x+1)
        # return (x - min(x)) / (max(x) - min(x))

def data_get(file_path, input_length, output_length, shuffle, batchsize,proportion):
    Bikeshare = pd.read_csv(file_path)
    Bikeshare.drop('instant', axis=1, inplace=True)  # 删除序号
    Bikeshare.drop('dteday', axis=1, inplace=True)  # 删除日期
    # Bikeshare.drop('atemp', axis=1, inplace=True)  # 删除日期
    Bikeshare.drop('casual', axis=1, inplace=True)  # 删除日期
    Bikeshare.drop('registered', axis=1, inplace=True)  # 删除日期

    Bikeshare = Bikeshare.dropna()
    Bikeshare = Bikeshare.apply(
        lambda x: minmax(x))

    # 将数据转换为序列
    def create_sequences(data, input_length, output_length):
        X, y = [], []
        for i in range(0,len(data) - input_length - output_length + 1,output_length):
            X.append(data[i:i + input_length, :-1])  # 使用除cnt之外的所有特征
            y.append(data[i + input_length:i + input_length + output_length, -1])  # cnt是最后一列
        return np.array(X), np.array(y)

    X, y = create_sequences(Bikeshare.values, input_length, output_length)

    # 划分训练集和测试集
    train_size = int(len(X) * proportion)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 转换为PyTorch张量
    X_train = torch.tensor(X, dtype=torch.float32)
    y_train = torch.tensor(y, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # 创建DataLoader
    train_dataset = BikeshareDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=shuffle)
    test_dataset = BikeshareDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)
    return train_loader,test_loader


class BikeshareDataset(Dataset):
    def __init__(self, X, y):
        self.x = X
        self.y = y

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)

if __name__ == '__main__':
    data_get('data/train_data.csv')
