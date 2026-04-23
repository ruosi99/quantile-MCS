import pandas as pd
import numpy as np
import copy
import torch
from torch.utils.data import Dataset
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,mean_absolute_percentage_error

from utils.model_training.dataset import EVDataset


def read_dataset():
    occ = pd.read_csv('datasets/occupancy.csv', index_col=0, header=0)
    inf = pd.read_csv('datasets/information.csv', index_col=None, header=0)
    prc = pd.read_csv('datasets/price.csv', index_col=0, header=0)
    adj = pd.read_csv('datasets/adj.csv', index_col=0, header=0)  # check
    dis = pd.read_csv('datasets/distance.csv', index_col=0, header=0)
    time = pd.read_csv('datasets/time.csv', index_col=None, header=0)

    col = occ.columns
    cap = np.array(inf['count'], dtype=float).reshape(1, -1)  # parking_capability
    occ = np.array(occ, dtype=float) / cap
    prc = np.array(prc, dtype=float)
    adj = np.array(adj, dtype=float)
    dis = np.array(dis, dtype=float)
    time = pd.to_datetime(time, dayfirst=True)
    return occ, prc, adj, col, dis, cap, time, inf

def read_dataset_v2(data_path='data/datasets/ST_EVCDP_v2_canonical/'):
    inf = pd.read_csv(data_path + 'inf.csv', index_col=None, header=0)
    inf.fillna(0, inplace=True)
    occ = pd.read_csv(data_path + 'occupancy.csv', index_col=0, header=0)
    occ.fillna(0, inplace=True)
    e_price = pd.read_csv(data_path + 'e_price.csv', index_col=0, header=0)
    e_price.fillna(0, inplace=True)
    s_price = pd.read_csv(data_path + 's_price.csv', index_col=0, header=0)
    s_price.fillna(0, inplace=True)
    price = e_price + s_price
    adj = pd.read_csv(data_path + 'adjacency_matrix.csv', index_col=0, header=0)  # check
    adj.fillna(0, inplace=True)
    duration = pd.read_csv(data_path + 'duration.csv', index_col=0, header=0)
    duration.fillna(0, inplace=True)
    cap = np.array(inf['pile_count'], dtype=float).reshape(1, -1)  # parking_capability

    occ = np.array(occ, dtype=float) / cap
    duration = np.array(duration, dtype=float) / cap
    price = np.array(price, dtype=float)
    adj = torch.tensor(np.array(adj, dtype=float), dtype=torch.float32)
    return occ, duration, price, adj, cap

def read_dataset_charged(data_path = 'data/datasets/charged/SZH/'):

    def fill_na(df):
        import numpy as np
        import pandas as pd
        if isinstance(df, pd.DataFrame) or isinstance(df, pd.Series):
            df.fillna(0, inplace=True)
        elif isinstance(df, np.ndarray):
            # NumPy 数组：用 np.nan_to_num 或布尔索引填充
            df = np.nan_to_num(df, nan=0.0)  # 返回新数组，不能 inplace
            # 或者：
            # df[np.isnan(df)] = 0
        else:
            raise TypeError("Unsupported type for fill_na")
        return df  # 注意：NumPy 情况不能 inplace，必须返回

    dataset = EVDataset(feature='volume', auxiliary='all', data_path=data_path)
    cap = np.array(dataset.sites['charger_num'], dtype=float).reshape(1, -1)
    volume = fill_na(dataset.volume) / cap
    duration = fill_na(dataset.duration) / cap
    e_price = fill_na(dataset.e_price)
    s_price = fill_na(dataset.s_price)
    price = e_price + s_price
    distance = fill_na(dataset.distance)
    time_info = dataset.time

    volume = np.array(volume, dtype=float)
    duration = np.array(duration, dtype=float)
    price = np.array(price, dtype=float)
    distance = torch.tensor(np.array(distance, dtype=float), dtype=torch.float32)

    return volume, duration, price, distance, time_info, cap

def read_dataset_as_df(data_path='data/datasets/ST_EVCDP_v2_canonical/'):
    inf = pd.read_csv(data_path + 'inf.csv', index_col=0, header=0)
    inf.fillna(0, inplace=True)
    occ = pd.read_csv(data_path + 'occupancy.csv', index_col=0, header=0)
    occ.fillna(0, inplace=True)
    e_price = pd.read_csv(data_path + 'e_price.csv', index_col=0, header=0)
    e_price.fillna(0, inplace=True)
    s_price = pd.read_csv(data_path + 's_price.csv', index_col=0, header=0)
    s_price.fillna(0, inplace=True)
    price = e_price + s_price
    adj = pd.read_csv(data_path + 'adjacency_matrix.csv', index_col=0, header=0)  # check
    adj.fillna(0, inplace=True)
    duration = pd.read_csv(data_path + 'duration.csv', index_col=0, header=0)
    duration.fillna(0, inplace=True)
    cap_np = np.array(inf['pile_count'], dtype=float).reshape(1, -1)  # parking_capability

    occ = occ.div(cap_np, axis=1)
    duration = duration.div(cap_np, axis=1)
    # price = np.array(price, dtype=float)
    # adj = np.array(adj, dtype=float)
    return occ, duration, price, adj, inf['pile_count']

# ---------data transform-----------
def create_rnn_data(dataset, lookback, predict_time):
    x = []
    y = []
    for i in range(len(dataset) - lookback - predict_time):
        x.append(dataset[i:i + lookback])
        y.append(dataset[i + lookback + predict_time - 1])
    return np.array(x), np.array(y)


def get_a_delta(adj):  # D^-1/2 * A * D^-1/2
    # adj.shape = np.size(node, node)
    deg = np.sum(adj, axis=0)
    deg = np.diag(deg)
    deg_delta = np.linalg.inv(np.sqrt(deg))
    a_delta = np.matmul(np.matmul(deg_delta, adj), deg_delta)
    return a_delta


def division(data, train_rate, valid_rate, calib_rate, test_rate):

    assert abs(train_rate + valid_rate + calib_rate + test_rate - 1) < 1e-6

    n = len(data)

    train_end = int(n * train_rate)
    valid_end = int(n * (train_rate + valid_rate))
    calib_end = int(n * (train_rate + valid_rate + calib_rate))

    train = data[:train_end]
    valid = data[train_end:valid_end]
    calib = data[valid_end:calib_end]
    test = data[calib_end:]

    return train, valid, calib, test


def set_seed(seed, flag):
    if flag == True:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def metrics(test_pre, test_real):
    eps = 0.01
    MAPE_test_real = test_real
    MAPE_test_pre = test_pre
    MAPE_test_real[np.where(MAPE_test_real == 0)] = MAPE_test_real[np.where(MAPE_test_real == 0)] + eps
    MAPE_test_pre[np.where(MAPE_test_real == 0)] = MAPE_test_pre[np.where(MAPE_test_real == 0)] + eps
    MAPE = mean_absolute_percentage_error(MAPE_test_real, MAPE_test_pre)
    MAE = mean_absolute_error(test_real, test_pre)
    MSE = mean_squared_error(test_real, test_pre)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(test_real, test_pre)
    RAE = np.sum(abs(test_pre - test_real)) / np.sum(abs(np.mean(test_real) - test_real))
    print('MAPE: {}'.format(MAPE))
    print('MAE:{}'.format(MAE))
    print('MSE:{}'.format(MSE))
    print('RMSE:{}'.format(RMSE))
    print('R2:{}'.format(R2))
    print(('RAE:{}'.format(RAE)))
    output_list = [MSE, RMSE, MAPE, RAE, MAE, R2]
    return output_list

import numpy as np

def metrics_new(test_pre, test_real):
    eps = 0.01  # 避免除零

    # 排查数据分布
    print("Test Real Min:", np.min(test_real), "Max:", np.max(test_real), "Mean:", np.mean(test_real))
    print("Test Pred Min:", np.min(test_pre), "Max:", np.max(test_pre), "Mean:", np.mean(test_pre))

    # 排查接近零的值
    near_zero_indices = np.where(test_real < 1e-8)
    if len(near_zero_indices[0]) > 0:
        print("Near-zero values in Test Real:", test_real[near_zero_indices])

    # 修正 MAPE 计算
    valid_indices = test_real > eps  # 忽略接近零的值
    valid_real = test_real[valid_indices]
    valid_pred = test_pre[valid_indices]
    MAPE = np.mean(np.abs((valid_real - valid_pred) / valid_real)) * 100

    # 其他指标
    MAE = np.mean(np.abs(test_real - test_pre))
    MSE = np.mean((test_real - test_pre) ** 2)
    RMSE = np.sqrt(MSE)

    # R²
    sst = np.sum((test_real - np.mean(test_real)) ** 2) + eps
    ssr = np.sum((test_real - test_pre) ** 2)
    R2 = 1 - (ssr / sst)

    # RAE
    RAE = np.sum(np.abs(test_pre - test_real)) / (np.sum(np.abs(test_real - np.mean(test_real))) + eps)

    # MedAE
    MedAE = np.median(np.abs(test_real - test_pre))

    # EVS (Explained Variance Score)
    diff = test_real - test_pre
    evs_num = np.var(diff)
    evs_den = np.var(test_real)
    EVS = 1 - (evs_num / (evs_den + eps))  # 加 eps 避免除零

    # 打印指标
    print('MAPE: {}'.format(MAPE))
    print('MAE: {}'.format(MAE))
    print('MSE: {}'.format(MSE))
    print('RMSE: {}'.format(RMSE))
    print('R²: {}'.format(R2))
    print('RAE: {}'.format(RAE))
    print('MedAE: {}'.format(MedAE))
    print('EVS: {}'.format(EVS))

    output_list = [MSE, RMSE, MAPE, RAE, MAE, R2, MedAE, EVS]
    return output_list

class CreateDataset(Dataset):
    def __init__(self, occ, prc, seq_l, pre_l, device):  # adj
        occ, label = create_rnn_data(occ, seq_l, pre_l)
        prc, _ = create_rnn_data(prc, seq_l, pre_l)
        self.occ = torch.Tensor(occ)
        self.prc = torch.Tensor(prc)
        self.label = torch.Tensor(label)
        self.device = device

    def __len__(self):
        return len(self.occ)

    def __getitem__(self, idx):  # occ: batch, seq, node
        output_occ = torch.transpose(self.occ[idx, :, :], 0, 1)
        output_prc = torch.transpose(self.prc[idx, :, :], 0, 1)
        output_label = self.label[idx, :]
        return output_occ, output_prc, output_label

def create_time_windows(time_index, seq_len, pred_len):
    """
    将一维时间索引转换为滑动窗口样本。
    返回: List[List[pd.Timestamp]]，长度 = 样本数
    """
    windows = []
    total_len = len(time_index)
    for i in range(total_len - seq_len - pred_len):
        window = time_index[i : i + seq_len].tolist()  # 历史部分，长度 seq_len
        windows.append(window)
    return windows

def batchify_time_windows(time_windows, batch_size, drop_last=True):
    """将 List[List[Timestamp]] 按 batch_size 分组"""
    batches = []
    n = len(time_windows)
    if drop_last:
        n = (n // batch_size) * batch_size
    for i in range(0, n, batch_size):
        batch = time_windows[i:i + batch_size]
        if len(batch) == batch_size or not drop_last:
            batches.append(batch)
    return batches


class CreateFastDataset(Dataset):
    def __init__(self, occ, prc, lb, pt, law, device, adj, num_layers=2, prob=0.6):  # adj
        occ, label = create_rnn_data(occ, lb, pt)
        prc, _ = create_rnn_data(prc, lb, pt)
        self.occ = torch.Tensor(occ)
        self.prc = torch.Tensor(prc)
        self.label = torch.Tensor(label)
        self.device = device
        self.adj = adj
        self.eye = torch.eye(adj.shape[0])
        self.deg = torch.sum(adj, dim=0)
        self.num_layers = num_layers
        self.law = -law

        # price
        chg = torch.randn(size=[self.occ.shape[2]]) / 2
        chg[torch.where(chg < prob)] = 0
        self.prc_chg = chg  # [node, ]

        # label
        chg = torch.unsqueeze(chg, dim=1)  # [node, 1]
        deg = torch.unsqueeze(self.deg, dim=1)  # [node, 1]
        label_chg = [-chg]
        hop_chg = chg
        for n in range(self.num_layers):  # graph propagation
            hop_chg = torch.matmul(self.adj-self.eye, hop_chg) * (1 / deg)
            label_chg.append(hop_chg)
        label_chg = torch.stack(label_chg, dim=1)  # [node, num_layers]
        label_chg = torch.sum(label_chg, dim=1)  # [node, ]
        self.label_chg = torch.squeeze(label_chg, dim=1)

    def __len__(self):
        return len(self.occ)

    def __getitem__(self, idx):  # occ: batch, seq, node
        # Pseudo Sampling
        prc_ch = torch.Tensor(self.prc[idx, :, :] * (1+self.prc_chg))  # [node, seq]
        label_ch = torch.tan(torch.Tensor(self.label[idx, :] * (1+self.label_chg/self.law)))  # [node, ]

        # to device
        output_occ = torch.transpose(self.occ[idx, :, :], 0, 1).to(self.device)
        output_prc = torch.transpose(self.prc[idx, :, :], 0, 1).to(self.device)
        output_label = self.label[idx, :].to(self.device)
        output_prc_ch = torch.transpose(prc_ch, 0, 1).to(self.device)
        output_label_ch = label_ch.to(self.device)
        return output_occ, output_prc, output_label, output_prc_ch, output_label_ch

class PseudoDataset(Dataset):
    def __init__(self, occ, prc, lb, pt, device,
                 adj_dense, law, num_layers=2,
                 prop=0.4, sigma=1.0, k=8, use_topk=True):   # ← 新增 k 和 use_topk
        occ, label = create_rnn_data(occ, lb, pt)
        prc, _ = create_rnn_data(prc, lb, pt)
        self.occ = torch.tensor(occ, dtype=torch.float32)
        self.prc = torch.tensor(prc, dtype=torch.float32)
        self.label = torch.tensor(label, dtype=torch.float32)
        self.device = device

        # === 距离转相似度 ===
        if not isinstance(adj_dense, torch.Tensor):
            adj_dense = torch.tensor(adj_dense, dtype=torch.float32)
        adj = torch.exp(-adj_dense / sigma)
        adj.fill_diagonal_(0)

        # === 稀疏化（Top-k） ===
        if use_topk:
            _, topk_idx = torch.topk(adj, k=k, dim=1, largest=True)
            mask = torch.zeros_like(adj)
            mask.scatter_(1, topk_idx, 1.0)
            adj = adj * mask

        self.adj = adj
        self.eye = torch.eye(adj.shape[0])
        self.deg = torch.sum(adj, dim=1)
        self.num_layers = num_layers
        self.prop = prop
        self.law = -law

        # === price change 与 label change 计算 ===
        node_score = torch.rand(size=[self.occ.shape[2]])
        shred = torch.quantile(node_score, self.prop)
        prc_chg = torch.randn_like(node_score) / 2
        prc_chg[torch.where(node_score > self.prop)] = 0
        self.prc_chg = prc_chg

        label_chg = self.law * prc_chg
        label_chg = label_chg.unsqueeze(1)
        hop_chg = -label_chg
        label_chg = [label_chg]
        deg = self.deg.unsqueeze(1)
        for n in range(self.num_layers):
            hop_chg = torch.matmul(self.adj - self.eye, hop_chg) * (1 / (deg + 1e-6))
            label_chg.append(hop_chg)
        label_chg = torch.stack(label_chg, dim=1).sum(dim=1)
        self.label_chg = label_chg.squeeze(1)

    def __getitem__(self, idx):
        pseudo_prc = self.prc[idx] * (1 + self.prc_chg)
        pseudo_label = torch.tan(self.label[idx] * (1 + self.label_chg))
        return (self.occ[idx].transpose(0, 1),
                self.prc[idx].transpose(0, 1),
                self.label[idx],
                pseudo_prc.transpose(0, 1),
                pseudo_label)

    def __len__(self):
        return len(self.occ)


def meta_division(data, support_rate, query_rate):
    data_length = len(data)
    support_division_index = int(data_length * support_rate)
    supprot_set = data[:support_division_index, :]
    query_set = data[support_division_index:, :]
    return supprot_set, query_set

def meta_time_division(data, support_rate, query_rate):
    data_length = len(data)
    support_division_index = int(data_length * support_rate)
    supprot_set = data[:support_division_index]
    query_set = data[support_division_index:]
    return supprot_set, query_set


def zero_init_global_gradient(model):
    grads = dict()
    for name, param in model.named_parameters():
        param.requires_grad_(True)
        grads[name] = 0
    return grads


def data_mix(ori_data, pse_data, mix_ratio):
    shred = int(ori_data.shape[0] * mix_ratio)
    mix_data = ori_data
    mix_data[shred:] = pse_data[shred:]  # mix on the 1st dimension: batch
    return mix_data

def build_time_marks_from_timestamps(
    ts_enc_batch, pred_len, label_len, device, *,
    mode: str = "float",  # "float" -> 归一化值, "long" -> 用作 embedding 索引
):
    """
    构造时间标记张量（用于 PAGInformer 或 Informer 模型）

    参数：
    --------
    ts_enc_batch : list[list[pd.Timestamp]]
        长度为 B 的列表，每个元素是长度为 T 的 pandas.Timestamp 序列（编码端历史时间戳）
    pred_len : int
        预测长度（模型最终输出步数）
    label_len : int
        解码器输入的历史长度
    device : torch.device
    mode : str, optional
        "float"  -> 归一化时间特征（默认，适合直接输入模型）
        "long"   -> 整数索引模式（适合后续 nn.Embedding）
    
    返回：
    --------
    mark_enc         [B, T, 4]
    mark_future_mid  [B, T, 4]
    mark_dec_final   [B, label_len + pred_len, 4]
    """
    B = len(ts_enc_batch)
    T = len(ts_enc_batch[0])
    assert all(len(seq) == T for seq in ts_enc_batch), "每个 batch 序列长度必须相同"

    # 判断时间粒度（小时 or 天）
    dt = ts_enc_batch[0][1] - ts_enc_batch[0][0]
    is_hourly = dt <= pd.Timedelta(hours=1.5)
    step = pd.Timedelta(hours=1) if is_hourly else pd.Timedelta(days=1)

    def row_from_ts(ts: pd.Timestamp):
        # 月(1-12), 日(1-31), 周(0-6), 时(0-23)
        m, d, w, h = ts.month, ts.day, ts.weekday(), ts.hour
        if mode == "float":
            return [
                (m - 1) / 11,      # month: 0~1
                (d - 1) / 30,      # day:   0~1
                w / 6,             # weekday: 0~1
                h / 23 if is_hourly else 0.0  # 如果是日粒度，不用 hour
            ]
        else:
            return [m, d, w, h]

    # 1️⃣ 编码端 mark_enc
    enc_list = [[row_from_ts(ts) for ts in seq] for seq in ts_enc_batch]

    # 2️⃣ 中间层 mark_future_mid（取未来 T 步）
    fut_mid_list = []
    for seq in ts_enc_batch:
        last = seq[-1]
        fut = [last + step * (i + 1) for i in range(T)]
        fut_mid_list.append([row_from_ts(ts) for ts in fut])

    # 3️⃣ 最终解码层 mark_dec_final（label_len + pred_len）
    dec_final_list = []
    for seq in ts_enc_batch:
        hist = seq[-label_len:]
        fut = [seq[-1] + step * (i + 1) for i in range(pred_len)]
        dec_final_list.append([row_from_ts(ts) for ts in hist + fut])

    # 4️⃣ 转 tensor
    dtype = torch.float if mode == "float" else torch.long
    mark_enc = torch.tensor(enc_list, dtype=dtype, device=device)
    mark_future_mid = torch.tensor(fut_mid_list, dtype=dtype, device=device)
    mark_dec_final = torch.tensor(dec_final_list, dtype=dtype, device=device)

    # 检查维度一致性
    assert mark_enc.shape == (B, T, 4)
    assert mark_future_mid.shape[0] == B
    assert mark_dec_final.shape[0] == B

    return mark_enc, mark_future_mid, mark_dec_final

def dense_to_sparse_adj(adj_dense: torch.Tensor, sigma: float = 1.0, k: int = 8) -> torch.Tensor:
    """
    Convert a dense adjacency matrix to a sparse adjacency matrix using Gaussian kernel
    and top-k sparsification.

    Steps:
        1. Apply Gaussian kernel: adj = exp(-adj_dense / sigma)
        2. Zero out diagonal (no self-loop)
        3. Keep only top-k largest values per row
        4. Return as a sparse COO tensor

    Args:
        adj_dense (torch.Tensor): Dense adjacency matrix of shape [N, N]
        sigma (float): Scaling factor for Gaussian kernel (default: 1.0)
        k (int): Number of neighbors to keep per node (default: 8)

    Returns:
        torch.Tensor: Sparse adjacency matrix in COO format (size [N, N])
    """
    assert adj_dense.dim() == 2 and adj_dense.shape[0] == adj_dense.shape[1], \
        "adj_dense must be a square 2D tensor"

    N = adj_dense.size(0)
    device = adj_dense.device

    # Step 1: Gaussian kernel
    adj = torch.exp(-adj_dense / sigma)

    # Step 2: Remove self-loops
    adj.fill_diagonal_(0)

    # Step 3: Keep top-k neighbors per row
    if k >= N:
        # If k >= N, keep all (but diagonal already zeroed)
        mask = (adj > 0).float()
    else:
        _, topk_idx = torch.topk(adj, k=k, dim=1, largest=True)  # [N, k]
        mask = torch.zeros_like(adj)
        mask.scatter_(1, topk_idx, 1.0)

    # Step 4: Apply mask
    adj = adj * mask

    # Step 5: Convert to sparse COO tensor
    adj_sparse = adj.to_sparse()

    return adj_sparse

def prepare_adj_dense_from_distance(
    dist_mat,
    method="gaussian",
    sigma=None,
    threshold=None,
    ensure_self_loop=True,
    force_symmetric=True,
    device=None
):
    """
    将原始距离矩阵转换为适用于 GCN 类模型的稠密邻接矩阵。
    ✅ 统一兼容你列出的所有图模型（GCN, LstmGcn, HSTGCN）
    ✅ 自动加自环（强烈推荐）
    ✅ 支持高斯核或二值化
    ✅ 防止归一化时出现除零错误

    参数:
        dist_mat: [N, N] 距离矩阵（Tensor 或 array，值 >= 0）
        method: "gaussian"（默认）或 "binary"
        sigma: 高斯核带宽（若为 None，自动设为非对角距离的均值）
        threshold: 二值化阈值（method="binary" 时使用）
        ensure_self_loop: 是否确保每个节点有自环（强烈建议 True）
        force_symmetric: 是否强制对称（建议 True）
        device: 输出设备（如 torch.device('cuda')）

    返回:
        adj_dense: [N, N] 邻接矩阵（Tensor），可直接传入模型的 adj_dense 参数
    """
    # 转为 float32 Tensor
    if not isinstance(dist_mat, torch.Tensor):
        dist_mat = torch.tensor(dist_mat, dtype=torch.float32)
    else:
        dist_mat = dist_mat.float()

    if device is not None:
        dist_mat = dist_mat.to(device)

    N = dist_mat.shape[0]
    assert dist_mat.shape == (N, N), f"输入必须是方阵，当前形状: {dist_mat.shape}"

    # 强制对称（避免有向距离导致问题）
    if force_symmetric:
        dist_mat = (dist_mat + dist_mat.T) / 2.0

    # 转换为相似性矩阵
    if method == "gaussian":
        if sigma is None:
            # 排除对角线（距离为0）计算 sigma
            mask = ~torch.eye(N, dtype=torch.bool, device=dist_mat.device)
            nonzero_dists = dist_mat[mask]
            sigma = nonzero_dists.mean().clamp(min=1e-6)
        adj = torch.exp(-dist_mat ** 2 / (2.0 * sigma ** 2))
    elif method == "binary":
        if threshold is None:
            threshold = dist_mat.mean()
        adj = (dist_mat <= threshold).float()
    else:
        raise ValueError("method 必须是 'gaussian' 或 'binary'")

    # 确保非负
    adj = torch.clamp(adj, min=0.0)

    # 👉 关键：确保自环存在（所有模型都需要！）
    if ensure_self_loop:
        adj = adj + torch.eye(N, device=adj.device)

    # 再次对称化（数值稳定）
    if force_symmetric:
        adj = (adj + adj.T) / 2.0

    return adj
