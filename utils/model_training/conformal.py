import numpy as np
import torch

def _quantile_index(quantiles, q):
    # 允许 float 微小误差
    for i, v in enumerate(quantiles):
        if abs(v - q) < 1e-9:
            return i
    raise ValueError(f"Quantile {q} not found in quantiles={quantiles}. "
                     f"Please include {q} in your quantile list.")

def conformal_cqr_calibrate(model, 
                            calib_loader, 
                            quantiles, 
                            delta=0.1, 
                            device="cpu", 
                            capacity=None,
                            node_variance=None,     # 新增
                            n_groups=3
                            ):
    """
    CQR calibration for pred_len=1 (single horizon).
    model output: (B, N, Q)
    label: (B, N)

    Returns:
        s_hat: scalar, used to expand intervals
    """
    model.eval()
    alpha = delta / 2.0
    li = _quantile_index(quantiles, alpha)         # lower quantile index
    ui = _quantile_index(quantiles, 1.0 - alpha)   # upper quantile index

    if node_variance is not None:
        node_variance = np.array(node_variance)
        bins = np.quantile(node_variance, np.linspace(0,1,n_groups+1))
        group_ids = np.digitize(node_variance, bins[1:-1])


    if node_variance is None:
        scores = []
    else:
        scores = [[] for _ in range(n_groups)]

    with torch.no_grad():
        for demand, price, label in calib_loader:
            demand = demand.to(device)
            price = price.to(device)
            label = label.to(device)  # (B,N)

            pred_q = model(demand, price)  # (B,N,Q)

            L_raw = pred_q[..., li]  # (B,N)
            U_raw = pred_q[..., ui]  # (B,N)

            if capacity is not None:
                capacity = torch.tensor(capacity, dtype=torch.float32).to(device)
                L_raw = L_raw * capacity
                U_raw = U_raw * capacity
                label = label * capacity

            # s_i = max(L - y, y - U)
            s = torch.maximum(L_raw - label, label - U_raw)  # (B,N)

            s_np = s.detach().cpu().numpy()  # (B,N)

            if node_variance is None:
                scores.append(s_np.reshape(-1))
            else:
                for g in range(n_groups):
                    node_mask = (group_ids == g)
                    if node_mask.sum() == 0:
                        continue
                    group_scores = s_np[:, node_mask]
                    scores[g].append(group_scores.reshape(-1))


    if node_variance is None:

        scores = np.concatenate(scores)
        scores = np.sort(scores)
        n = scores.size

        k = int(np.ceil((n + 1) * (1 - delta))) - 1
        k = min(max(k, 0), n - 1)

        s_hat = float(scores[k])
        return s_hat

    else:

        s_hat = np.zeros(n_groups)

        for g in range(n_groups):
            if len(scores[g]) == 0:
                continue

            group_scores = np.concatenate(scores[g])
            group_scores = np.sort(group_scores)
            n = group_scores.size

            k = int(np.ceil((n + 1) * (1 - delta))) - 1
            k = min(max(k, 0), n - 1)

            s_hat[g] = group_scores[k]

        return s_hat, group_ids

def apply_cqr_interval(pred_q, quantiles, s_hat, delta=0.1,
                       nonneg=True,
                       group_ids=None):
    """
    pred_q: numpy array (T, N, Q) or (B, N, Q)
    returns:
        point_pred: (T,N) median q0.5
        L, U: calibrated interval (T,N)
    """
    s_hat = s_hat * 0
    alpha = delta / 2.0
    li = _quantile_index(quantiles, alpha)
    ui = _quantile_index(quantiles, 1.0 - alpha)
    mi = _quantile_index(quantiles, 0.5)

    L_raw = pred_q[..., li]
    U_raw = pred_q[..., ui]
    point = pred_q[..., mi]

    if group_ids is None:
        L = L_raw - s_hat
        U = U_raw + s_hat
    else:
        # group-wise
        L = np.zeros_like(L_raw)
        U = np.zeros_like(U_raw)

        for g in range(len(s_hat)):
            node_mask = (group_ids == g)
            if node_mask.sum() == 0:
                continue

            L[:, node_mask] = L_raw[:, node_mask] - s_hat[g]
            U[:, node_mask] = U_raw[:, node_mask] + s_hat[g]


    if nonneg:
        L = np.maximum(L, 0.0)
        U = np.maximum(U, 0.0)
        point = np.maximum(point, 0.0)

    return point, L, U

def interval_metrics(y_true, L, U, point_pred=None, delta=0.1):
    """
    y_true, L, U: numpy arrays (T,N)
    point_pred: numpy array (T,N), median prediction (optional, needed for WIS)
    delta: miscoverage rate (e.g., 0.1 for 90% coverage interval)
    
    returns:
        PICP: coverage
        MPIW: mean interval width
        WIS: Weighted Interval Score (requires point_pred)
    """
    covered = (y_true >= L) & (y_true <= U)
    picp = covered.mean()
    mpiw = (U - L).mean()
    
    metrics = {"PICP": float(picp), "MPIW": float(mpiw)}

    if point_pred is not None:
        if not 0.0 < delta < 1.0:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        alpha = delta / 2.0
        
        # 计算分位数损失
        # Lower Quantile Loss: rho_alpha(y, L)
        diff_L = L - y_true
        loss_L = np.maximum((1 - alpha) * diff_L, -alpha * diff_L)
        
        # Upper Quantile Loss: rho_{1-alpha}(y, U)
        # 这里的 alpha 对应下界分位数，上界分位数为 1-alpha
        # 损失函数形式：rho_tau(y, q) = (y-q)(tau - I(y<q))
        # 对于上界 tau = 1-alpha:
        # 如果 y < U: loss = (y-U)*(-alpha) = alpha*(U-y)
        # 如果 y >= U: loss = (y-U)*(1-alpha)
        diff_U = U - y_true
        loss_U = np.maximum(alpha * diff_U, -(1 - alpha) * diff_U)
        
        # Median Quantile Loss (MAE scaled): rho_0.5(y, point)
        # 这就是 0.5 * |y - point|
        diff_M = point_pred - y_true
        loss_M = np.maximum(0.5 * diff_M, -0.5 * diff_M)
        
        # Normalized WIS for one central interval plus the predictive median
        # is the sum of these three pinball losses divided by K + 0.5 = 1.5.
        wis = (loss_L + loss_U + loss_M).mean() / 1.5
        
        metrics["WIS"] = float(wis)
    
    return metrics
