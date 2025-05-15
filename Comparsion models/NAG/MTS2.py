import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import assessment
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from lion_pytorch import Lion
import logging
from collections import OrderedDict
from copy import deepcopy
'''在0的基础上加上温度，工作日因素'''
'''效果变好'''
torch.manual_seed(88)
import time
start = time.perf_counter()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_logger = logging.getLogger(__name__)

'''数据提取'''
data = pd.read_excel(r"C:\Users\11860\Desktop\论文1\数据\预测数据.xlsx", sheet_name="比利时")
df = data[["AVGV","温度","工作日"]].values.astype('float32')  # 转换数据类型为float32  684

look_back = 168

# list1 = df[0: int(len(df)*0.6)]#(5256, 1)

trainlist1 = df[0: int(len(df)*0.8)] #7008
validlist1 = df[int(len(df)*0.8)-look_back:int(len(df)*0.9)]#1044
testlist1 = df[int(len(df)*0.9)-look_back:]#1044

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(trainlist1)
valid_scaled = scaler.transform(validlist1)
test_scaled = scaler.transform(testlist1)

'''定义混合训练集处理函数'''
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(look_back,len(dataset)):
        load_dataX = dataset[i - look_back: i, :]
        load_dataY = dataset[i, 0]
        dataX.append(load_dataX)
        dataY.append(load_dataY)
    return np.array(dataX), np.array(dataY)

inputT, outputT = create_dataset(train_scaled, look_back)
inputV, outputV = create_dataset(valid_scaled, look_back)  # torch.Size([37, 81]),torch.Size([37, 1])
inputTEST, outputTEST = create_dataset(test_scaled, look_back)

# 转换为PyTorch张量
inputT = torch.tensor(inputT, dtype=torch.float32).to(device)
outputT = torch.tensor(outputT, dtype=torch.float32).to(device)
inputV = torch.tensor(inputV, dtype=torch.float32).to(device)
outputV = torch.tensor(outputV, dtype=torch.float32).to(device)
inputTEST = torch.tensor(inputTEST, dtype=torch.float32).to(device)
outputTEST = torch.tensor(outputTEST, dtype=torch.float32).to(device)

# outputT = scaler.inverse_transform(outputT.cpu().numpy())
# outputV = scaler.inverse_transform(outputV.cpu().numpy())
# outputTEST = scaler.inverse_transform(outputTEST.cpu().numpy())

'''定义网络架构'''
class MLPBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, input_dim)

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


class FactorizedTemporalMixing(nn.Module):
    def __init__(self, input_dim, mlp_dim, sampling):
        super().__init__()

        assert sampling in [1, 2, 3, 4, 6, 8, 12]
        self.sampling = sampling
        self.temporal_fac = nn.ModuleList([
            MLPBlock(input_dim // sampling, mlp_dim) for _ in range(sampling)
        ])

    def merge(self, shape, x_list):
        y = torch.zeros(shape, device=x_list[0].device)
        for idx, x_pad in enumerate(x_list):
            y[:, :, idx::self.sampling] = x_pad

        return y

    def forward(self, x):
        x_samp = []
        for idx, samp in enumerate(self.temporal_fac):
            x_samp.append(samp(x[:, :, idx::self.sampling]))

        x = self.merge(x.shape, x_samp)

        return x


class FactorizedChannelMixing(nn.Module):
    def __init__(self, input_dim, factorized_dim):
        super().__init__()

        assert input_dim <= factorized_dim
        self.channel_mixing = MLPBlock(input_dim, factorized_dim)

    def forward(self, x):
        return self.channel_mixing(x)


class MixerBlock(nn.Module):
    def __init__(self, tokens_dim, channels_dim, tokens_hidden_dim, channels_hidden_dim, fac_T, fac_C, sampling,
                 norm_flag):
        super().__init__()
        self.tokens_mixing = FactorizedTemporalMixing(tokens_dim, tokens_hidden_dim, sampling) if fac_T else MLPBlock(
            tokens_dim, tokens_hidden_dim)
        self.channels_mixing = FactorizedChannelMixing(channels_dim, channels_hidden_dim) if fac_C else None
        self.norm = nn.LayerNorm(channels_dim) if norm_flag else None

    def forward(self, x):
        y = self.norm(x) if self.norm else x
        y = self.tokens_mixing(y.transpose(1, 2)).transpose(1, 2)

        if self.channels_mixing:
            y += x
            res = y
            y = self.norm(y) if self.norm else y
            y = res + self.channels_mixing(y)

        return y


class ChannelProjection(nn.Module):
    def __init__(self, seq_length, pred_length, enc_in, individual):
        super().__init__()
        self.projection = nn.Linear(seq_length, pred_length)
        self.individual = individual
        if self.individual:
            self.projection = nn.ModuleList([nn.Linear(seq_length, pred_length) for _ in range(enc_in)])

    def forward(self, x):
        if self.individual:
            x = torch.stack([proj(x[:, i, :]) for i, proj in enumerate(self.projection)], dim=1)
        else:
            x = self.projection(x.transpose(1, 2)).transpose(1, 2)
        return x


class RevIN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.epsilon = 1e-6
        self.affine = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode='norm'):
        if mode == 'norm':
            mean = x.mean(dim=(0, 1), keepdim=True)
            std = x.std(dim=(0, 1), keepdim=True)
            x = (x - mean) / (std + self.epsilon)
            x = x * self.affine + self.bias
        elif mode == 'denorm':
            x = (x - self.bias) / (self.affine + self.epsilon)
            mean = x.mean(dim=(0, 1), keepdim=True)
            std = x.std(dim=(0, 1), keepdim=True)
            x = x * (std + self.epsilon) + mean
        return x


class Model(nn.Module):
    def __init__(self, seq_len, enc_in, pred_len, d_model, d_ff, e_layers, fac_T, fac_C, sampling, norm, rev):
        super().__init__()
        self.mlp_blocks = nn.ModuleList([
            MixerBlock(seq_len, enc_in, d_model, d_ff, fac_T, fac_C, sampling, norm) for _ in range(e_layers)
        ])
        self.norm = nn.LayerNorm(enc_in) if norm else None
        self.projection = ChannelProjection(seq_len, pred_len, enc_in, False)
        self.rev = RevIN(enc_in) if rev else None

    def forward(self, x):
        x = self.rev(x, 'norm') if self.rev else x

        for block in self.mlp_blocks:
            x = block(x)

        x = self.norm(x) if self.norm else x
        x = self.projection(x)
        x = self.rev(x, 'denorm') if self.rev else x

        return x[:,-1,:]


# 定义模型参数
seq_len = look_back
enc_in = 3  # 输入特征数量
pred_len = 1  # 预测的时间步
d_model = 20  # token混合层隐藏维度
d_ff = 20  # 通道混合层隐藏维度
e_layers = 4  # Mixer Block数量
fac_T = True  # 使用Factorized Temporal Mixing
fac_C = True  # 使用Factorized Channel Mixing
sampling = 3  # 采样比例
norm = False  # 使用Layer Norm
rev = False # 不使用RevIN

# 定义模型
model = Model(seq_len, enc_in, pred_len, d_model, d_ff, e_layers, fac_T, fac_C, sampling, norm, rev).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.004, momentum=0.97, nesterov=True)#lr=0.0032

# 训练模型68 0.95 660
num_epochs = 400#1206
train_losses = []  # 用于存储每个epoch的训练损失值
valid_losses = []  # 用于存储每个epoch的验证损失值

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(inputT)
    loss = criterion(outputs[:, 0], outputT)
    loss.backward()
    optimizer.step()

    # 验证损失计算
    model.eval()
    with torch.no_grad():
        valid_outputs = model(inputV)
        valid_loss = criterion(valid_outputs[:, 0], outputV).item()
        valid_losses.append(valid_loss)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Training Loss: {loss * 10000:.4f}*10^-4, '
              f'Validation Loss: {valid_loss * 10000:.4f}*10^-4, ')

print("---------模型训练完成，接下来开始预测---------")


'''预测'''
with torch.no_grad():
    model.eval()
    pre_train = model(inputT).cpu().numpy()
    pre_valid = model(inputV).cpu().numpy()
    pre_test = model(inputTEST).cpu().numpy()


# 反归一化并评估
def trans(a,origin_ndim):
    if isinstance(a, torch.Tensor) and a.is_cuda:
        a = a.detach().cpu().reshape(-1, 1).numpy()  # 将CUDA张量移动到CPU上并转换为NumPy数组
    if isinstance(a, np.ndarray):
        if a.ndim == origin_ndim:
            return scaler.inverse_transform(a)[:, 0]
        if a.ndim == 2 and a.shape[1] == 1: #是二维数组，且第二个维度，即列维度为1
            a = np.concatenate((a, np.zeros((a.shape[0], 2))), axis=1)  # 如果是一维数组，添加一个新的轴
        elif a.shape[1] == 2:
            a = np.concatenate((a, np.zeros((a.shape[0], 1))), axis=1)
    a = scaler.inverse_transform(a)[:, 0]  # 反归一化
    return a
origin_ndim = 3

pre_train = trans(pre_train,origin_ndim)
pre_valid = trans(pre_valid,origin_ndim)
pre_test = trans(pre_test,origin_ndim)

outputT = trans(outputT,3)
outputV = trans(outputV,3)
outputTEST = trans(outputTEST,3)

# for value in pre_end_ema:
#     print(value)

# 计算评价指标
results = {
    "MTS2 Training": [assessment.RMSE(outputT, pre_train), assessment.MAE(outputT, pre_train), assessment.MAPE(outputT, pre_train), assessment.R2(outputT, pre_train)],
    "MTS2 Validation": [assessment.RMSE(outputV, pre_valid), assessment.MAE(outputV, pre_valid), assessment.MAPE(outputV, pre_valid), assessment.R2(outputV, pre_valid)],
    "MTS2 Test": [assessment.RMSE(outputTEST, pre_test), assessment.MAE(outputTEST, pre_test), assessment.MAPE(outputTEST, pre_test), assessment.R2(outputTEST, pre_test)],
    }

# 转换为DataFrame并转置
df_results = pd.DataFrame(results, index=["RMSE", "MAE", "MAPE", "R2"]).T

# 打印表格
print(df_results)
end = time.perf_counter()
print("运行时间:", end - start)
for i in range(len(pre_test)):
    print(pre_test[i])
# 可视化预测集
# plt.rcParams['font.family'] = 'SimHei'  # 防止图像中文乱码
# trainY_x = torch.arange(1, len(pre_train) + 1, 1)
# plt.figure(dpi=500, figsize=(15, 5))
# plt.plot(trainY_x, trainY, color='b', label='训练集')
# plt.plot(trainY_x, pre_train, color='g', label='训练集预测结果')
# plt.xlabel(u'时间')
# plt.ylabel(u'流量')
# plt.show()

# plt.rcParams['font.family'] = 'SimHei'  # 防止图像中文乱码
# validY_x = torch.arange(0, len(trainY), 1)
# validY_x1 = torch.arange(len(trainY), len(pre_valid1) + len(trainY), 1)
# validY_x2 = torch.arange(len(pre_valid1) + len(trainY), len(pre_valid1) + len(trainY) + len(pre_valid2), 1)
# plt.figure(dpi=500, figsize=(15, 5))
# plt.plot(validY_x, trainY, color='b', label='训练期洪水过程')
# plt.plot(validY_x1, validY1, color='y', label='验证集洪水过程')
# plt.plot(validY_x2, validY2, color='y')
# plt.xlabel(u'历时(小时)')
# plt.ylabel(u'流量(立方米/秒)')
# plt.legend()
# plt.show()

# plt.rcParams['font.family'] = 'SimHei'  # 防止图像中文乱码
# validY_x1 = torch.arange(1, len(pre_valid1) + 1, 1)
# validY_x2 = torch.arange(len(pre_valid1) + 1, len(pre_valid1) + len(pre_valid2) + 1, 1)
# plt.figure(dpi=500, figsize=(15, 5))
# plt.plot(validY_x1, validY1, color='b', label='第一场洪水验证集')
# plt.plot(validY_x2, validY2, color='b', label='第二场洪水验证集')
# plt.plot(validY_x1, pre_valid1, color='g', label='第一场洪水验证集预测结果')
# plt.plot(validY_x2, pre_valid2, color='g', label='第二场洪水验证集预测结果')
# plt.xlabel(u'时间')
# plt.ylabel(u'流量')
# plt.legend()
# plt.show()

# plt.rcParams['font.family'] = 'SimHei'  # 防止图像中文乱码
# validY_x = torch.arange(1, len(pre_test) + 1, 1)
# plt.figure(dpi=500, figsize=(15, 5))
# plt.plot(validY_x, outputTEST, color='b', label='测试集真实值')
# plt.plot(validY_x, pre_test, color='g', label='测试集预测值')
# plt.xlabel(u'时间')
# plt.ylabel(u'负荷')
# plt.legend()
# plt.show()
#
# # 可视化训练和验证损失
# plt.figure(dpi=700,figsize=(12, 6))
# plt.plot([x for x in train_losses], label='Training Loss',linewidth=1)
# plt.plot([x for x in valid_losses], label='Validation Loss',linewidth=1)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.title('Training and Validation Loss vs EMA Loss')
# plt.show()
#
