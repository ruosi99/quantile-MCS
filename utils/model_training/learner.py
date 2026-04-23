import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import utils.model_training.training_utils as fn
import copy
from tqdm import tqdm
import os
from utils.model_training.loss_functions import QuantileLoss

def physics_informed_meta_learning(
    law_list, global_model, model_name, p_epoch, bs, 
    train_demand, train_price,
    seq_l, pre_l, device, distance, label_len = 24, out_dir = 'data/results/checkpoints', 
    quantiles = [0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95]
):
    os.makedirs(out_dir, exist_ok=True)
    support_demand, query_demand = fn.meta_division(train_demand, support_rate=0.5, query_rate=0.5)
    support_prc, query_prc = fn.meta_division(train_price, support_rate=0.5, query_rate=0.5)

    # 预训练数据生成
    n_laws = len(law_list)
    support_dataset_dict, query_dataset_dict = {}, {}
    support_dataloader_dict, query_dataloader_dict = {}, {}

    for n in tqdm(range(n_laws), desc="Preparing DataLoaders"):
        support_dataset_dict[n] = fn.PseudoDataset(support_demand, support_prc, seq_l, pre_l, device, distance, law_list[n])
        query_dataset_dict[n] = fn.PseudoDataset(query_demand, query_prc, seq_l, pre_l, device, distance, law_list[n])
        support_dataloader_dict[n] = DataLoader(support_dataset_dict[n], batch_size=bs, shuffle=False, drop_last=True)
        query_dataloader_dict[n] = DataLoader(query_dataset_dict[n], batch_size=bs, shuffle=False, drop_last=False)  # batch_size 改为 bs
    # 注意：support_loader 用了 drop_last=True，query 用了 False

    # Meta-learning 训练过程
    torch.save(global_model, f'{out_dir}/meta_{model_name}_{pre_l}_bs{bs}model.pt')
    loss_function = torch.nn.MSELoss() if "quantile" not in model_name else QuantileLoss(quantiles=quantiles)
    loss_function = loss_function.to(device)

    global_model.train()
    for epoch in tqdm(range(p_epoch), desc='Pre-training', position=0, leave=True):
        query_loss = float("inf")  # 初始化 query_loss
        global_grads = fn.zero_init_global_gradient(global_model)

        # Inner-loop 训练每个物理法则
        for n in tqdm(range(n_laws), desc=f'Epoch {epoch+1}/{p_epoch} - Training Laws', position=1, leave=False):
            temp_model = torch.load(f'{out_dir}/meta_{model_name}_{pre_l}_bs{bs}model.pt', weights_only=False).to(device)
            temp_optimizer = torch.optim.Adam(temp_model.parameters(), weight_decay=1e-5)
            temp_model.train()

            # 训练 Support Set
            for j, data in tqdm(enumerate(support_dataloader_dict[n]), total=len(support_dataloader_dict[n]),
                                desc=f'Law {n+1}/{n_laws} - Support', position=2, leave=False):
                demand, price, label, pseudo_price, pseudo_label = data
                demand, price, label, pseudo_price, pseudo_label = demand.to(device), price.to(device), label.to(device), pseudo_price.to(device), pseudo_label.to(device)

                mix_ratio = (j+1) * demand.shape[0] / len(support_demand)
                mix_prc = fn.data_mix(price, pseudo_price, mix_ratio)
                mix_label = fn.data_mix(label, pseudo_label, mix_ratio)

                temp_optimizer.zero_grad()
                predict = temp_model(demand, mix_prc)
                loss = loss_function(predict, mix_label)
                loss.backward()
                temp_optimizer.step()

                # 清理缓存
                del demand, price, label, pseudo_price, pseudo_label, mix_prc, mix_label, predict
                torch.cuda.empty_cache()

            # 训练 Query Set
            for j, data in tqdm(enumerate(query_dataloader_dict[n]), total=len(query_dataloader_dict[n]),
                                desc=f'Law {n+1}/{n_laws} - Query', position=3, leave=False):
                demand, price, label, pseudo_price, pseudo_label = data
                demand, price, label, pseudo_price, pseudo_label = demand.to(device), price.to(device), label.to(device), pseudo_price.to(device), pseudo_label.to(device)

                temp_optimizer.zero_grad()
                predict = temp_model(demand, price)
                    
                loss = loss_function(predict, label)
                loss.backward()

                for name, param in temp_model.named_parameters():
                    if param.grad is not None:
                        global_grads[name] += param.grad

        # Global 更新: BGD
        for name, param in global_model.named_parameters():
            param = param - 0.02 * global_grads[name] / n_laws

        # 保存最佳模型
        if loss < query_loss:
            query_loss = loss
            torch.save(global_model, f'{out_dir}/meta_{model_name}_{pre_l}_bs{bs}model.pt')

    return global_model


def fast_learning(law_list, model, model_name, p_epoch, bs, train_occupancy, train_price, seq_l, pre_l, device, adj_dense):
    n_laws = len(law_list)
    fast_datasets = dict()
    fast_loaders = dict()
    for n in range(n_laws):
        fast_datasets[n] = fn.CreateFastDataset(train_occupancy, train_price, seq_l, pre_l, law_list[n], device, adj_dense)
        fast_loaders[n] = DataLoader(fast_datasets[n], batch_size=bs, shuffle=True, drop_last=True)
    
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.00001)
    loss_function = torch.nn.MSELoss()
    for epoch in tqdm(range(p_epoch), desc='Pre-training'):
        for n in range(n_laws):
            for j, data in enumerate(fast_loaders[n]):
                '''
                occupancy = (batch, seq, node)
                price = (batch, seq, node)
                label = (batch, node)
                '''
                occupancy, price, label, prc_ch, label_ch = data
                optimizer.zero_grad()
                predict = model(occupancy, prc_ch)
                loss = loss_function(predict, label_ch)
                loss.backward()
                optimizer.step()

            for j, data in enumerate(fast_loaders[n]):
                '''
                occupancy = (batch, seq, node)
                price = (batch, seq, node)
                label = (batch, node)
                '''
                occupancy, price, label, prc_ch, label_ch = data
                optimizer.zero_grad()
                predict = model(occupancy, prc_ch)
                loss = loss_function(predict, label_ch)
                loss.backward()
                optimizer.step()

    return model
