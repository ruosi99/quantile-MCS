import numpy as np
import matplotlib.pyplot as plt

def plot_comparison(predict_list, label_list, index):
    """
    根据给定的索引提取预测值和真实值，并绘制对比图。
    
    Parameters:
    predict_list (numpy.ndarray): 预测值数组，形状为 (时间, 区域)。
    label_list (numpy.ndarray): 真实值数组，形状为 (时间, 区域)。
    index (int): 要提取的区域索引。
    
    """
    # 提取对应区域的所有数据
    predict_values = predict_list[:, index]
    label_values = label_list[:, index]
    print(label_values)

    # 绘制对比图
    plt.figure(figsize=(12, 6))
    plt.plot(predict_values, label='Predict', color='blue')
    plt.plot(label_values, label='Real', color='green')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'Comparison of Predictions and Real Values for Region {index}')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # 加载 predict_list.npy 和 label_list.npy 文件
    predict_list_loaded = np.load('data/occ_pred_res/predict_list.npy')
    label_list_loaded = np.load('data/occ_pred_res/label_list.npy')

    # 打印加载的数据以确认
    print(predict_list_loaded.shape)
    print(label_list_loaded.shape)

    plot_comparison(predict_list_loaded, label_list_loaded, 46)
