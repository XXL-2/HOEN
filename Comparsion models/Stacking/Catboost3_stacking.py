import catboost as cb
import pandas as pd
import numpy as np
from invoke import GRU_i2
from invoke import LSTM_i2
from invoke import MTS_i2
import assessment
import time
start = time.perf_counter()
'''传统STACKING,采用k折交叉验证'''
'''训练并预测LSTM'''
lstm_pre_valid, lstm_pre_valid0, lstm_pre_test, outputT_oooorgin, outputV_oooorgin, outputTEST_orgin = LSTM_i2.LSTM(load_model_flag=False)
print('~~~~~~~~~~~~LSTM训练预测完成~~~~~~~~~~~~')

'''训练并预测GRU'''
gru_pre_valid, gru_pre_valid0, gru_pre_test,_, _, _ = GRU_i2.GRU(load_model_flag=False)
print('~~~~~~~~~~~~GRU训练预测完成~~~~~~~~~~~~')

'''训练并预测MTS'''
mts_pre_valid, mts_pre_valid0, mts_pre_test,_, _, _ = MTS_i2.MTS(load_model_flag=False)
print('~~~~~~~~~~~~MTS训练预测完成~~~~~~~~~~~~')

# 将LSTM、GRU和MTS的预测结果作为特征，使用CatBoost进行集成
X_train_meta = np.hstack((lstm_pre_valid.reshape(-1, 1), gru_pre_valid.reshape(-1, 1), mts_pre_valid.reshape(-1, 1)))
X_valid_meta = np.hstack((lstm_pre_valid0.reshape(-1, 1), gru_pre_valid0.reshape(-1, 1), mts_pre_valid0.reshape(-1, 1)))
X_test_meta = np.hstack((lstm_pre_test.reshape(-1, 1), gru_pre_test.reshape(-1, 1), mts_pre_test.reshape(-1, 1)))

y_train_meta = outputT_oooorgin
y_valid_meta = outputV_oooorgin

# 设置CatBoost的参数
params = {
    'loss_function': 'RMSE',  # 使用回归任务
    'eval_metric': 'RMSE',    # 使用RMSE作为评估指标
    'learning_rate': 0.04,
    'depth': 6,
    'iterations': 1000,#45
    'early_stopping_rounds': 100,
    'verbose': 100
}

# 创建CatBoost Pool数据集
train_pool = cb.Pool(X_train_meta, y_train_meta)
valid_pool = cb.Pool(X_valid_meta, y_valid_meta)

# 训练CatBoost模型
print('Training CatBoost model...')
model = cb.CatBoostRegressor(**params)
model.fit(train_pool, eval_set=valid_pool)

# 在训练集、验证集和测试集上进行预测
y_pred_train = model.predict(X_train_meta)
y_pred_valid = model.predict(X_valid_meta)
y_pred_test = model.predict(X_test_meta)

# 计算评估指标
results = {
    "CatBoost-Stacking Training": [assessment.RMSE(y_train_meta, y_pred_train), assessment.MAE(y_train_meta, y_pred_train),
                          assessment.MAPE(y_train_meta, y_pred_train), assessment.R2(y_train_meta, y_pred_train)],
    "CatBoost-Stacking Validation": [assessment.RMSE(y_valid_meta, y_pred_valid), assessment.MAE(y_valid_meta, y_pred_valid),
                            assessment.MAPE(y_valid_meta, y_pred_valid), assessment.R2(y_valid_meta, y_pred_valid)],
    "CatBoost-Stacking Test": [assessment.RMSE(outputTEST_orgin, y_pred_test), assessment.MAE(outputTEST_orgin, y_pred_test),
                      assessment.MAPE(outputTEST_orgin, y_pred_test), assessment.R2(outputTEST_orgin, y_pred_test)],
}

# 将结果存储为DataFrame并输出
df_results_cb = pd.DataFrame(results, index=["RMSE", "MAE", "MAPE", "R2"]).T
print(df_results_cb)
end = time.perf_counter()
print("运行时间:", end - start)
for i in range(len(y_pred_test)):
    print(y_pred_test[i])