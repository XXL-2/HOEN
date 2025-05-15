import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import assessment
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import logging
from collections import OrderedDict
from copy import deepcopy
'''在0的基础上加上温度，工作日因素'''
'''效果变好'''
import time
start = time.perf_counter()

torch.manual_seed(88)

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
print(outputT.shape)
print(outputV.shape)
# outputT = scaler.inverse_transform(outputT.cpu().numpy())
# outputV = scaler.inverse_transform(outputV.cpu().numpy())
# outputTEST = scaler.inverse_transform(outputTEST.cpu().numpy())

'''定义网络架构'''
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        self.lstm1.flatten_parameters()  # 添加这一行来解决权重警告
        out, _ = self.lstm1(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # 取最后一个时间步的输出作为输入到全连接层
        out = self.fc(out)
        return out

# 定义模型
model = LSTMModel(input_size=3, hidden_size=20, output_size=1)
model.to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.009)

# 训练模型68 0.95 660
num_epochs = 400#1206
train_losses = []  # 用于存储每个epoch的训练损失值
valid_losses = []  # 用于存储每个epoch的验证损失值

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(inputT)
    loss = criterion(outputs.squeeze(), outputT)
    loss.backward()
    optimizer.step()

    # 验证损失计算
    model.eval()
    with torch.no_grad():
        valid_outputs = model(inputV)
        valid_loss = criterion(valid_outputs.squeeze(), outputV).item()
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
    "LSTM Training": [assessment.RMSE(outputT, pre_train), assessment.MAE(outputT, pre_train), assessment.MAPE(outputT, pre_train), assessment.R2(outputT, pre_train)],
    "LSTM Validation": [assessment.RMSE(outputV, pre_valid), assessment.MAE(outputV, pre_valid), assessment.MAPE(outputV, pre_valid), assessment.R2(outputV, pre_valid)],
    "LSTM Test": [assessment.RMSE(outputTEST, pre_test), assessment.MAE(outputTEST, pre_test), assessment.MAPE(outputTEST, pre_test), assessment.R2(outputTEST, pre_test)],
    }

# 转换为DataFrame并转置
df_results = pd.DataFrame(results, index=["RMSE", "MAE", "MAPE", "R2"]).T

# 打印表格
print(df_results)
end = time.perf_counter()
print("运行时间:", end - start)
# for i in range(len(pre_valid)):
#     print(pre_valid[i])
# print('')
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

plt.rcParams['font.family'] = 'SimHei'  # 防止图像中文乱码
validY_x = torch.arange(1, len(pre_test) + 1, 1)
plt.figure(dpi=500, figsize=(15, 5))
plt.plot(validY_x, outputTEST, color='b', label='测试集真实值')
plt.plot(validY_x, pre_test, color='g', label='测试集预测值')
plt.xlabel(u'时间')
plt.ylabel(u'负荷')
plt.legend()
plt.show()

# 可视化训练和验证损失
plt.figure(dpi=700,figsize=(12, 6))
plt.plot([x for x in train_losses], label='Training Loss',linewidth=1)
plt.plot([x for x in valid_losses], label='Validation Loss',linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss vs EMA Loss')
plt.show()

