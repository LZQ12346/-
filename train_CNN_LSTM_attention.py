import torch
from torch import nn
from dataset import data_get
from model import CNN_LSTM_Attention
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

train_datapath = '/home/dzhang/wudan/machine_learning/LSTM/data/train_data.csv'
test_datapath = '/home/dzhang/wudan/machine_learning/LSTM/data/test_data.csv'
outputpath='/home/dzhang/wudan/machine_learning/LSTM/output/model_cnn_lstm_attention_96.pt'
input_size = 12
input_length = 96
output_length = 240
hidden_size = 64
num_layers = 2
dropout = 0.2
lr = 1e-3
batchsize = 256
epochs = 300
device = torch.device('cuda:9' if torch.cuda.is_available() else 'cpu')



def train(outputpath):
    train_dataloder,test_dataloder = data_get(train_datapath, input_length, output_length, True, batchsize,1)
    test_dataloder,_ = data_get(test_datapath, input_length, output_length, False, batchsize,1)
    model = CNN_LSTM_Attention(input_size, hidden_size, output_length, num_layers, dropout)
    model.to(device)
    loss_fn = nn.MSELoss()
    mse_min=np.inf
    mae_min=np.inf
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss=0
        mse=0
        mae=0
        for i, (data, label) in enumerate(train_dataloder):
            x = data.to(device)
            y = label.to(device)
            pre = model(x)
            loss = loss_fn(pre, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # mae += mean_absolute_error(pre.cpu().detach().numpy()*(cnt_max - cnt_min)+cnt_min, y.cpu().detach().numpy()*(cnt_max - cnt_min)+cnt_min)
            # mse+=mean_squared_error(pre.cpu().detach().numpy()*(cnt_max - cnt_min)+cnt_min, y.cpu().detach().numpy()*(cnt_max - cnt_min)+cnt_min)

            mae += mean_absolute_error(pre.cpu().detach().numpy(), y.cpu().detach().numpy())
            mse+=mean_squared_error(pre.cpu().detach().numpy(), y.cpu().detach().numpy())

            total_loss+=loss.item()
        # print('epoch:', epoch, 'loss:', total_loss/len(train_dataloder), 'mae:', mae/len(train_dataloder), 'mse:', mse/len(train_dataloder))
        model.eval()
        Tmae=0
        Tmse=0
        for i, (data, label) in enumerate(test_dataloder):
            x = data.to(device)
            y = label.to(device)
            pre = model(x)
            # Tmae += mean_absolute_error(pre.cpu().detach().numpy()*(cnt_max - cnt_min)+cnt_min, y.cpu().detach().numpy()*(cnt_max - cnt_min)+cnt_min)
            # Tmse+=mean_squared_error(pre.cpu().detach().numpy()*(cnt_max - cnt_min)+cnt_min, y.cpu().detach().numpy()*(cnt_max - cnt_min)+cnt_min)
            Tmae+= mean_absolute_error(np.exp(pre.cpu().detach().numpy()), np.exp(y.cpu().detach().numpy()))
            Tmse+=mean_squared_error(np.exp(pre.cpu().detach().numpy()), np.exp(y.cpu().detach().numpy()))
            # Tmae+= mean_absolute_error(pre.cpu().detach().numpy(), y.cpu().detach().numpy())
            # Tmse+=mean_squared_error(pre.cpu().detach().numpy(), y.cpu().detach().numpy())
        print('epoch:', epoch, 'loss:', total_loss/len(train_dataloder), 'mae:', mae/len(train_dataloder), 'mse:', mse/len(train_dataloder),'Tmae:', Tmae/len(test_dataloder), 'Tmse:', Tmse/len(test_dataloder))
        if Tmse<mse_min:
            mse_min=Tmse
            mae_min=Tmae
            torch.save(model.state_dict(),outputpath)
            print('save')
    return mse_min,mae_min

mse_min_list=[]
mae_min_list=[]
for k in range(5):
    outputpath='/home/dzhang/wudan/machine_learning/LSTM/output/model_cnn_lstm_atten_{output_length}_{k}.pt'
    outputpath=outputpath.format(k=k,output_length=output_length)
    mse_min,mae_min=train(outputpath)
    mae_min_list.append(mae_min)
    mse_min_list.append(mse_min)
for j in range(5):
    print(mae_min_list[j],mse_min_list[j])