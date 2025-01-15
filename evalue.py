import torch
from torch import nn
import numpy as np
import pandas as pd
from dataset_test import data_get
from model import LSTMModel,BikeShareTransformer,BikeShaareTransformerall,LSTM_with_Attention,CNN_LSTM_Attention
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

train_datapath = '/home/dzhang/wudan/machine_learning/LSTM/data/train_data.csv'
test_datapath = '/home/dzhang/wudan/machine_learning/LSTM/data/test_data.csv'
outputpath='/home/dzhang/wudan/machine_learning/LSTM/output/model_LSTM_96.pt'
input_size = 12
d_model = 128
nhead = 4
input_length = 96
# output_length = 96
output_length = 240
hidden_size = 64
num_layers = 4
dropout = 0.2
lr = 1e-3
batchsize = 1
epochs = 300
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def test(outputpath,model):
    test_dataloder,_ = data_get(test_datapath, input_length, output_length, False, batchsize,1)
    
    model.load_state_dict(torch.load(outputpath))
    model.to(device)
    model.eval()
    mae=0
    mse=0
    pres=[]
    real_label=[]
    for i, (data, label) in enumerate(test_dataloder):
        x = data.to(device)
        y = label.to(device)
        # pre = model(x)[-1,:,:]
        # pre = model(x)
        pre = model(x,x)
        # pre = model(x)
        # pre = model(x)
        mae+= mean_absolute_error(np.exp(pre.cpu().detach().numpy()), np.exp(y.cpu().detach().numpy()))
        mse+=mean_squared_error(np.exp(pre.cpu().detach().numpy()), np.exp(y.cpu().detach().numpy()))
        pres.extend(np.exp(pre.cpu().detach().numpy()).tolist()[0])
        real_label.extend(np.exp(y.cpu().detach().numpy()).tolist()[0])
    print('mae:', mae/len(test_dataloder), 'mse:', mse/len(test_dataloder))

    sns.set(style='whitegrid', palette='tab10')  # 设置样式和调色板
    # 创建画布
    plt.figure(figsize=(10, 6))

    x=[i for i in range(len(pres))]
    lens=500
    x=x[:lens]
    pres=pres[:lens]
    real_label=real_label[:lens]
    # 绘制第一个折线图
    # sns.lineplot(x=x, y=pres, label='lstm-96', linewidth=1)
    # sns.lineplot(x=x, y=pres, label='tranformer-encoder-96', linewidth=1)
    # sns.lineplot(x=x, y=pres, label='tranformer-all-96', linewidth=1)
    # sns.lineplot(x=x, y=pres, label='lstm-attention-96', linewidth=1)
    # sns.lineplot(x=x, y=pres, label='cnn-lstm-attention-96', linewidth=1)

    # sns.lineplot(x=x, y=pres, label='lstm-240', linewidth=1)
    # sns.lineplot(x=x, y=pres, label='tranformer-encoder-240', linewidth=1)
    sns.lineplot(x=x, y=pres, label='tranformer-all-240', linewidth=1)
    # sns.lineplot(x=x, y=pres, label='lstm-attention-240', linewidth=1)
    # sns.lineplot(x=x, y=pres, label='cnn-lstm-attention-240', linewidth=1)

    # 绘制第二个折线图
    sns.lineplot(x=x, y=real_label, label='real', linewidth=1)

    # 设置标题和标签
    plt.title('Comparison chart between actual value and predicted value')
    plt.xlabel('time sequence')
    plt.ylabel('cnt')

    # 显示图例
    plt.legend()

    # 显示图形
    plt.savefig(outputpath.replace('.pt','.jpg').replace('/output/','/test_output/'),dpi=300, transparent=True)
    datas=pd.DataFrame([pres,real_label]).T
    datas.to_csv(outputpath.replace('.pt','.csv').replace('/output/','/excel_out/'))
    return mse/len(test_dataloder),mae/len(test_dataloder)

mse_min_list=[]
mae_min_list=[]
for k in range(5):
    # model = LSTMModel(input_size, hidden_size, output_length, num_layers, dropout)
    # model = BikeShareTransformer(input_size, d_model, nhead, num_layers,output_length,dropout)
    model = BikeShaareTransformerall(input_size, d_model, nhead, num_layers,output_length,dropout)
    # model = LSTM_with_Attention(input_size, hidden_size, output_length, num_layers, dropout)
    # model = CNN_LSTM_Attention(input_size, hidden_size, output_length, num_layers, dropout)

    # outputpath='/home/dzhang/wudan/machine_learning/LSTM/output/model_LSTM_96_{k}.pt'
    # outputpath='/home/dzhang/wudan/machine_learning/LSTM/output/model_trans_encode_96_{k}.pt'
    # outputpath='/home/dzhang/wudan/machine_learning/LSTM/output/model_trans_all_96_{k}.pt'
    # outputpath='/home/dzhang/wudan/machine_learning/LSTM/output/model_lstm_atten_96_{k}.pt'
    # outputpath='/home/dzhang/wudan/machine_learning/LSTM/output/model_cnn_lstm_atten_96_{k}.pt'

    # outputpath='/home/dzhang/wudan/machine_learning/LSTM/output/model_LSTM_240_{k}.pt'
    # outputpath='/home/dzhang/wudan/machine_learning/LSTM/output/model_trans_encode_240_{k}.pt'
    outputpath='/home/dzhang/wudan/machine_learning/LSTM/output/model_trans_all_240_{k}.pt'
    # outputpath='/home/dzhang/wudan/machine_learning/LSTM/output/model_lstm_atten_240_{k}.pt'
    # outputpath='/home/dzhang/wudan/machine_learning/LSTM/output/model_cnn_lstm_atten_240_{k}.pt'
    outputpath=outputpath.format(k=k)
    mse_min,mae_min=test(outputpath,model)
    mae_min_list.append(mae_min)
    mse_min_list.append(mse_min)
for j in range(5):
    print(mae_min_list[j])
print()
for j in range(5):
    print(mse_min_list[j])