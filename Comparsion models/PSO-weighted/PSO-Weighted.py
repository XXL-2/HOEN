import numpy as np
from pyswarm import pso
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score
import assessment
from invoke import LSTM_i
from invoke import GRU_i
from invoke import MTS_i
import time
start = time.perf_counter()
# 三个子模型的预测结果和真实标签
'''训练并预测LSTM'''
lstm_pre_train, lstm_pre_valid, lstm_pre_test, outputT, outputV, outputTEST, _, _, _ \
    = LSTM_i.LSTM(load_model_flag=True)
print('~~~~~~~~~~~~LSTM训练预测完成~~~~~~~~~~~~')

'''训练并预测GRU'''
gru_pre_train, gru_pre_valid, gru_pre_test, _, _, _, trainlist, validlist, testlist \
    = GRU_i.GRU(load_model_flag=True)
print('~~~~~~~~~~~~GRU训练预测完成~~~~~~~~~~~~')

'''训练并预测MTS'''
mts_pre_train, mts_pre_valid, mts_pre_test, _, _, _, _, _, _, \
    = MTS_i.MTS(load_model_flag=True)
print('~~~~~~~~~~~~MTS训练预测完成~~~~~~~~~~~~')

y_true = outputV  # 真实标签
model_1_pred = lstm_pre_valid  # 模型1预测值
model_2_pred = gru_pre_valid  # 模型2预测值
model_3_pred = mts_pre_valid  # 模型3预测值

# 损失函数，可以使用 MSE 作为优化目标
def objective_function(weights):
    # 归一化权重
    weights = weights / np.sum(weights)

    # 计算加权预测
    weighted_pred2 = weights[0] * lstm_pre_valid + weights[1] * gru_pre_valid + weights[2] * mts_pre_valid
    mse_loss2 = assessment.RMSE(outputV, weighted_pred2)
    # 计算加权 MSE 损失
    mse_loss = mse_loss2
    return mse_loss



# PSO 优化的上下界，确保权重为正
lb = [0, 0, 0]
ub = [1, 1, 1]

# 运行PSO，找到最佳权重组合
best_weights, _ = pso(objective_function, lb, ub,
                      swarmsize=100,       # 粒子数，设置为30
                      maxiter=1000,        # 最大迭代次数，设置为100
                      omega=0.1,          # 惯性权重
                      phip=0.1,           # 个体最佳位置的加速度系数
                      phig=0.2)           # 全局最佳位置的加速度系数

# 输出最优权重
best_weights = best_weights / np.sum(best_weights)  # 归一化
print("最佳权重组合:", best_weights)

# 计算最终预测结果
valid_predictions = best_weights[0] * model_1_pred + best_weights[1] * model_2_pred + best_weights[2] * model_3_pred
test_predictions = best_weights[0] * lstm_pre_test + best_weights[1] * gru_pre_test + best_weights[2] * mts_pre_test
train_predictions = best_weights[0] * lstm_pre_train + best_weights[1] * gru_pre_train + best_weights[2] * mts_pre_train
# print("最终预测结果:", final_prediction)
results = {
    "PSO-weight Validation": [assessment.RMSE(outputV, valid_predictions), assessment.MAE(outputV, valid_predictions),
                           assessment.MAPE(outputV, valid_predictions), assessment.R2(outputV, valid_predictions)],
    "PSO-weight Test": [assessment.RMSE(outputTEST, test_predictions), assessment.MAE(outputTEST, test_predictions),
                     assessment.MAPE(outputTEST, test_predictions), assessment.R2(outputTEST, test_predictions)],
}
df_results_swa = pd.DataFrame(results, index=["RMSE", "MAE", "MAPE", "R2"]).T
print(df_results_swa)
end = time.perf_counter()
print("运行时间:", end - start)
# for i in range(len(train_predictions)):
#     print(train_predictions[i])
# print('')
# for i in range(len(valid_predictions)):
#     print(valid_predictions[i])
# print('')
for i in range(len(test_predictions)):
    print(test_predictions[i])