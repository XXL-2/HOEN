import catboost as cb
import pandas as pd
import numpy as np
from invoke import LightGBM_i
from invoke import XGBoost_i
import assessment
import time
from sklearn.neural_network import MLPRegressor
start = time.perf_counter()
'''传统STACKING,采用k折交叉验证'''
'''训练并预测LGBM'''
lgbm_pre_valid, lgbm_pre_valid0, lgbm_pre_test, outputT_oooorgin, outputV_oooorgin, outputTEST_orgin = LightGBM_i.LGBM(load_model_flag=False)
print('~~~~~~~~~~~~LGBM训练预测完成~~~~~~~~~~~~')

'''训练并预测XGB'''
xgb_pre_valid, xgb_pre_valid0, xgb_pre_test,_, _, _ = XGBoost_i.XGBM(load_model_flag=False)
print('~~~~~~~~~~~~XGB训练预测完成~~~~~~~~~~~~')

# 将LSTM、GRU和MTS的预测结果作为特征，使用CatBoost进行集成
X_train_meta = np.hstack((lgbm_pre_valid.reshape(-1, 1), xgb_pre_valid.reshape(-1, 1)))
X_valid_meta = np.hstack((lgbm_pre_valid0.reshape(-1, 1), xgb_pre_valid0.reshape(-1, 1)))
X_test_meta = np.hstack((lgbm_pre_test.reshape(-1, 1), xgb_pre_test.reshape(-1, 1)))

y_train_meta = outputT_oooorgin
y_valid_meta = outputV_oooorgin

# 创建MLPRegressor模型
model = MLPRegressor(
    hidden_layer_sizes=(8, 8),  # 设置隐藏层的神经元数目
    solver='adam',                  # 使用Adam优化器
    learning_rate='constant',  # 使用固定学习率
    learning_rate_init=0.0001,  # 设置初始学习率
    max_iter=200,                  # 最大迭代次数
    early_stopping=True,            # 启用早停
    random_state=88
)
# 训练MLP模型
print('Training MLP model...')
model.fit(X_train_meta, y_train_meta)
print("MLP模型实际迭代次数:", model.n_iter_)
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