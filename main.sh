#!/bin/bash

# 使脚本具有可执行权限
chmod +x main.py

# 运行Python脚本，并提供所需的命令行参数

# =========================不同模型预测的初始mcs进行分配的结果===========================

python main.py \
        --stations_file data/busiest_cluster/busiest_cluster_stations.csv \
        --mcs_file data/mcs_info/mcs_info_300_no_pred.csv \
        --ev_request_file data/new_ev_requests/ev_requests_scale_5.csv \
        --output_dir no_pred \
        --algorithm greedy_algorithm

python main.py \
        --stations_file data/busiest_cluster/busiest_cluster_stations.csv \
        --mcs_file data/mcs_info/mcs_info_by_model_fgn_300.csv \
        --ev_request_file data/new_ev_requests/ev_requests_scale_5.csv \
        --output_dir with_pred_fgn \
        --algorithm greedy_algorithm

python main.py \
        --stations_file data/busiest_cluster/busiest_cluster_stations.csv \
        --mcs_file data/mcs_info/mcs_info_by_model_lstm_300.csv \
        --ev_request_file data/new_ev_requests/ev_requests_scale_5.csv \
        --output_dir with_pred_lstm \
        --algorithm greedy_algorithm

python main.py \
        --stations_file data/busiest_cluster/busiest_cluster_stations.csv \
        --mcs_file data/mcs_info/mcs_info_by_model_fgn_300.csv \
        --ev_request_file data/new_ev_requests/ev_requests_scale_5.csv \
        --output_dir with_pred_fgn \
        --algorithm greedy_algorithm

python main.py \
        --stations_file data/busiest_cluster/busiest_cluster_stations.csv \
        --mcs_file data/mcs_info/mcs_info_by_model_pag_300.csv \
        --ev_request_file data/new_ev_requests/ev_requests_scale_5.csv \
        --output_dir with_pred_pag \
        --algorithm greedy_algorithm

python main.py \
        --stations_file data/busiest_cluster/busiest_cluster_stations.csv \
        --mcs_file data/mcs_info/mcs_info_by_model_pag_informer_best_300.csv \
        --ev_request_file data/new_ev_requests/ev_requests_scale_5.csv \
        --output_dir with_pred_pag_informer \
        --algorithm greedy_algorithm

# =========================MCS数量的敏感性分析===========================

python main.py \
        --stations_file data/busiest_cluster/busiest_cluster_stations.csv \
        --mcs_file data/mcs_info/mcs_info_by_model_pag_informer_best_50.csv \
        --ev_request_file data/new_ev_requests/ev_requests_scale_3.csv \
        --output_dir with_pred_pag_informer_num_50 \
        --algorithm greedy_algorithm

python main.py \
        --stations_file data/busiest_cluster/busiest_cluster_stations.csv \
        --mcs_file data/mcs_info/mcs_info_by_model_pag_informer_best_100.csv \
        --ev_request_file data/new_ev_requests/ev_requests_scale_3.csv \
        --output_dir with_pred_pag_informer_num_100 \
        --algorithm greedy_algorithm

python main.py \
        --stations_file data/busiest_cluster/busiest_cluster_stations.csv \
        --mcs_file data/mcs_info/mcs_info_by_model_pag_informer_best_150.csv \
        --ev_request_file data/new_ev_requests/ev_requests_scale_3.csv \
        --output_dir with_pred_pag_informer_num_150 \
        --algorithm greedy_algorithm

python main.py \
        --stations_file data/busiest_cluster/busiest_cluster_stations.csv \
        --mcs_file data/mcs_info/mcs_info_by_model_pag_informer_best_200.csv \
        --ev_request_file data/new_ev_requests/ev_requests_scale_3.csv \
        --output_dir with_pred_pag_informer_num_200 \
        --algorithm greedy_algorithm

python main.py \
        --stations_file data/busiest_cluster/busiest_cluster_stations.csv \
        --mcs_file data/mcs_info/mcs_info_by_model_pag_informer_best_250.csv \
        --ev_request_file data/new_ev_requests/ev_requests_scale_3.csv \
        --output_dir with_pred_pag_informer_num_250 \
        --algorithm greedy_algorithm

python main.py \
        --stations_file data/busiest_cluster/busiest_cluster_stations.csv \
        --mcs_file data/mcs_info/mcs_info_by_model_pag_informer_best_300.csv \
        --ev_request_file data/new_ev_requests/ev_requests_scale_3.csv \
        --output_dir with_pred_pag_informer_num_300 \
        --algorithm greedy_algorithm

python main.py \
        --stations_file data/busiest_cluster/busiest_cluster_stations.csv \
        --mcs_file data/mcs_info/mcs_info_by_model_pag_informer_best_350.csv \
        --ev_request_file data/new_ev_requests/ev_requests_scale_3.csv \
        --output_dir with_pred_pag_informer_num_350 \
        --algorithm greedy_algorithm

python main.py \
        --stations_file data/busiest_cluster/busiest_cluster_stations.csv \
        --mcs_file data/mcs_info/mcs_info_by_model_pag_informer_best_400.csv \
        --ev_request_file data/new_ev_requests/ev_requests_scale_3.csv \
        --output_dir with_pred_pag_informer_num_400 \
        --algorithm greedy_algorithm

python main.py \
        --stations_file data/busiest_cluster/busiest_cluster_stations.csv \
        --mcs_file data/mcs_info/mcs_info_by_model_pag_informer_best_450.csv \
        --ev_request_file data/new_ev_requests/ev_requests_scale_3.csv \
        --output_dir with_pred_pag_informer_num_450 \
        --algorithm greedy_algorithm

python main.py \
        --stations_file data/busiest_cluster/busiest_cluster_stations.csv \
        --mcs_file data/mcs_info/mcs_info_by_model_pag_informer_best_500.csv \
        --ev_request_file data/new_ev_requests/ev_requests_scale_3.csv \
        --output_dir with_pred_pag_informer_num_500 \
        --algorithm greedy_algorithm

# =========================噪声水平的敏感性分析===========================

python main.py \
        --stations_file data/busiest_cluster/busiest_cluster_stations.csv \
        --mcs_file data/mcs_info/mcs_info_pag_informer_best_noise_0.00_300.csv \
        --ev_request_file data/new_ev_requests/ev_requests_scale_3.csv \
        --output_dir with_pred_pag_informer_noise_0.00 \
        --algorithm greedy_algorithm

python main.py \
        --stations_file data/busiest_cluster/busiest_cluster_stations.csv \
        --mcs_file data/mcs_info/mcs_info_pag_informer_best_noise_0.05_300.csv \
        --ev_request_file data/new_ev_requests/ev_requests_scale_3.csv \
        --output_dir with_pred_pag_informer_noise_0.05 \
        --algorithm greedy_algorithm

python main.py \
        --stations_file data/busiest_cluster/busiest_cluster_stations.csv \
        --mcs_file data/mcs_info/mcs_info_pag_informer_best_noise_0.10_300.csv \
        --ev_request_file data/new_ev_requests/ev_requests_scale_3.csv \
        --output_dir with_pred_pag_informer_noise_0.10 \
        --algorithm greedy_algorithm

python main.py \
        --stations_file data/busiest_cluster/busiest_cluster_stations.csv \
        --mcs_file data/mcs_info/mcs_info_pag_informer_best_noise_0.15_300.csv \
        --ev_request_file data/new_ev_requests/ev_requests_scale_3.csv \
        --output_dir with_pred_pag_informer_noise_0.15 \
        --algorithm greedy_algorithm

python main.py \
        --stations_file data/busiest_cluster/busiest_cluster_stations.csv \
        --mcs_file data/mcs_info/mcs_info_pag_informer_best_noise_0.20_300.csv \
        --ev_request_file data/new_ev_requests/ev_requests_scale_3.csv \
        --output_dir with_pred_pag_informer_noise_0.20 \
        --algorithm greedy_algorithm

python main.py \
        --stations_file data/busiest_cluster/busiest_cluster_stations.csv \
        --mcs_file data/mcs_info/mcs_info_pag_informer_best_noise_0.25_300.csv \
        --ev_request_file data/new_ev_requests/ev_requests_scale_3.csv \
        --output_dir with_pred_pag_informer_noise_0.25 \
        --algorithm greedy_algorithm

python main.py \
        --stations_file data/busiest_cluster/busiest_cluster_stations.csv \
        --mcs_file data/mcs_info/mcs_info_pag_informer_best_noise_0.30_300.csv \
        --ev_request_file data/new_ev_requests/ev_requests_scale_3.csv \
        --output_dir with_pred_pag_informer_noise_0.30 \
        --algorithm greedy_algorithm

# =========================移动充电偏好敏感性分析===========================

# =========================gamma敏感性分析===========================