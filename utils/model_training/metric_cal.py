import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 加载 npy 文件
predict_list = np.load('data/occ_pred_res/predict_list.npy')
label_list = np.load('data/occ_pred_res/label_list.npy')

# 初始化误差列表
mse_list = []
rmse_list = []
mae_list = []
r2_list = []

# 按列计算误差
for i in range(predict_list.shape[1]):
    mse = mean_squared_error(label_list[:, i], predict_list[:, i])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(label_list[:, i], predict_list[:, i])  # 确保传入两个参数
    r2 = r2_score(label_list[:, i], predict_list[:, i])
    
    mse_list.append(mse)
    rmse_list.append(rmse)
    mae_list.append(mae)
    r2_list.append(r2)

# 计算平均误差
mean_mse = np.mean(mse_list)
mean_rmse = np.mean(rmse_list)
mean_mae = np.mean(mae_list)
mean_r2 = np.mean(r2_list)

print(f"Mean Squared Error (MSE): {mean_mse}")
print(f"Root Mean Squared Error (RMSE): {mean_rmse}")
print(f"Mean Absolute Error (MAE): {mean_mae}")
print(f"R² Score: {mean_r2}")
