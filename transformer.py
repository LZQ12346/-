import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_length, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class BikeShareTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_seq_len):
        super().__init__()
        
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Linear(d_model, 1)
        self.output_seq_len = output_seq_len

    def forward(self, src, src_mask=None):
        # src shape: [batch_size, seq_len, features]
        
        # Embedding and positional encoding
        x = self.input_embedding(src)
        x = self.positional_encoding(x)
        
        # Transformer encoder
        if src_mask is None:
            src_mask = self._generate_square_subsequent_mask(src.size(1))
        
        output = self.transformer_encoder(x, src_mask)
        
        # Predict next values
        predictions = self.output_layer(output)
        return predictions[:, -self.output_seq_len:, :]  # Return only the predicted sequence

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_window=96, output_window=96):
        self.data = data
        self.input_window = input_window
        self.output_window = output_window
        
    def __len__(self):
        return len(self.data) - self.input_window - self.output_window + 1
    
    def __getitem__(self, idx):
        input_data = self.data[idx:idx + self.input_window]
        target_data = self.data[idx + self.input_window:idx + self.input_window + self.output_window, 0]  # Only cnt column
        return torch.FloatTensor(input_data), torch.FloatTensor(target_data)

def prepare_data(df):
    # 将分类变量转换为独热编码
    categorical_cols = ['season', 'yr', 'holiday', 'workingday', 'weathersit']
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    
    # 标准化数值特征
    numerical_cols = ['temp', 'atemp', 'hum', 'windspeed']
    scaler = StandardScaler()
    df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
    
    # 确保cnt列在最前面
    cols = df_encoded.columns.tolist()
    cols.remove('cnt')
    cols = ['cnt'] + cols
    df_encoded = df_encoded[cols]
    
    return df_encoded.values

def train_model(model, train_loader, val_loader, num_epochs, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output.squeeze(-1), batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model(batch_x)
                val_loss += criterion(output.squeeze(-1), batch_y).item()
        
        print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'bike_share_transformer_{"short" if model.output_seq_len == 96 else "long"}.pth')

def main():
    # 假设数据已经加载到df中
    # df = pd.read_csv('bike_sharing_data.csv')
    
    # 数据预处理
    processed_data = prepare_data(df)
    
    # 创建数据集
    # 短期预测 (96小时)
    short_term_dataset = TimeSeriesDataset(processed_data, input_window=96, output_window=96)
    # 长期预测 (240小时)
    long_term_dataset = TimeSeriesDataset(processed_data, input_window=96, output_window=240)
    
    # 模型参数
    input_dim = processed_data.shape[1]  # 特征数量
    d_model = 128
    nhead = 8
    num_layers = 4
    
    # 创建短期预测模型
    short_term_model = BikeShareTransformer(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        output_seq_len=96
    )
    
    # 创建长期预测模型
    long_term_model = BikeShareTransformer(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        output_seq_len=240
    )
    
    # 训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    short_term_model.to(device)
    long_term_model.to(device)
    
    # 训练短期预测模型
    train_model(short_term_model, short_term_train_loader, short_term_val_loader, num_epochs=50, device=device)
    
    # 训练长期预测模型
    train_model(long_term_model, long_term_train_loader, long_term_val_loader, num_epochs=50, device=device)

if __name__ == '__main__':
    main()