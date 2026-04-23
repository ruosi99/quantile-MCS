import numpy as np
import pandas as pd
from utils.model_training.training_utils import metrics_new

def calculate_original_metrics(city, variable, model):
    base_path = f"data/results/{city}/{variable}_{model}_on_pretrain_results"
    label_path = f"{base_path}/label_list.npy"
    predict_path = f"{base_path}/predict_list.npy"
    
    label = np.load(label_path)
    predict = np.load(predict_path)
    
    sites_path = f"data/datasets/charged/{city}/sites.csv"
    volume_path = f"data/datasets/charged/{city}/volume.csv"

    df = pd.read_csv(sites_path, header=0, index_col=0)
    volume_df = pd.read_csv(volume_path, header=0, index_col=0)
    df = df.reindex(pd.to_numeric(volume_df.columns, errors='raise'))

    charger_nums = df['charger_num'].values
    
    if len(charger_nums) != label.shape[1]:
        raise ValueError(f"Mismatch in number of stations: {len(charger_nums)} vs {label.shape[1]}")
    
    label_original = label * charger_nums[None, :]
    predict_original = predict * charger_nums[None, :]
    
    metrics = metrics_new(predict_original, label_original)
    
    # 打印指标以便查看
    print(f"Metrics for {city} {variable} {model}:")
    metric_names = ['MSE', 'RMSE', 'MAPE', 'RAE', 'MAE', 'R2']
    for name, value in zip(metric_names, metrics):
        print(f"{name}: {value}")
    
    return metrics

if __name__ == "__main__":
    # 新增主流程
    import pandas as pd
    
    cities = ['AMS', 'JHB', 'LOA', 'MEL', 'SPO', 'SZH']
    models = ['gcn', 'lstm_gat', 'lstm_gcn', 'var', 'pag', 'pag_s1', 'pag_s2', 'lstm']
    variable = 'volume'
    
    results = {}
    metric_names = ['MSE', 'RMSE', 'MAPE', 'RAE', 'MAE', 'R2']
    
    for city in cities:
        results[city] = []
        for model in models:
            try:
                metrics = calculate_original_metrics(city, variable, model)
                metrics_dict = dict(zip(metric_names, metrics))
                metrics_dict['Model'] = model
                results[city].append(metrics_dict)
            except FileNotFoundError:
                print(f"Skipping {city} {variable} {model}: Files not found.")
            except Exception as e:
                print(f"Error for {city} {variable} {model}: {e}")
    
    # 保存到 Excel
    output_file = 'data/results/metrics_summary.xlsx'
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        # 先创建一个空 sheet 以避免 KeyError
        pd.DataFrame().to_excel(writer, sheet_name='Summary', index=False, header=False)
        
        worksheet = writer.sheets['Summary']  # 现在可以安全获取
        
        startrow = 0
        for city, city_results in results.items():
            if city_results:
                # 先写入城市标题
                worksheet.write(startrow, 0, f"City: {city}")
                startrow += 2  # 加1行标题后加1空行
                
                df = pd.DataFrame(city_results)
                df = df.sort_values(by='MAE', ascending=True)
                df.to_excel(writer, sheet_name='Summary', startrow=startrow, index=False)
                
                # 更新 startrow: header (1) + data rows (len(df)) + 1 额外空行分隔
                startrow += 1 + len(df) + 1
