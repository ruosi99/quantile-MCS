import math
import torch
import torch.nn as nn
from torch_geometric.utils import to_undirected
from torch_scatter import scatter_softmax, scatter_add
import torch.nn.functional as F
import copy


import utils.model_training.training_utils as fn
from utils.informer.model import Informer as Informer

use_cuda = True
device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
fn.set_seed(seed=2023, flag=True)

class LSTM(nn.Module):
    def __init__(self, seq, n_fea, node=247):
        super(LSTM, self).__init__()
        self.nodes = node
        self.encoder = nn.Conv2d(self.nodes, self.nodes, (n_fea, n_fea))  # input.shape: [batch, channel, width, height]
        self.lstm = nn.LSTM(self.nodes, self.nodes, num_layers=2, batch_first=True)
        self.decoder = nn.Linear(seq-n_fea+1, 1)

    def forward(self, occ, prc):  # occ.shape = [batch, node, seq]
        x = torch.stack([occ, prc], dim=3)
        x = self.encoder(x)
        x = torch.transpose(x.squeeze(), 1, 2)  # shape [batch, seq-n_fea+1, node]
        x, _ = self.lstm(x)
        x = torch.transpose(x, 1, 2)  # shape [batch, node, seq-n_fea+1]
        x = self.decoder(x)
        x = torch.squeeze(x)
        return x

class MultiHeadsGATLayer(nn.Module):
    def __init__(self, a_sparse, input_dim, out_dim, head_n, dropout, alpha, train_heads=True):  # input_dim = seq_length
        super(MultiHeadsGATLayer, self).__init__()

        self.head_n = head_n
        self.train_heads = train_heads
        self.head_weights = nn.ParameterList()
        self.head_attn = nn.ParameterList()
        param_device = a_sparse.device
        for n in range(head_n):
            weight = nn.Parameter(torch.empty(size=(input_dim, out_dim), device=param_device))
            attn = nn.Parameter(torch.empty(size=(1, 2 * out_dim), device=param_device))
            nn.init.xavier_normal_(weight, gain=1.414)
            nn.init.xavier_normal_(attn, gain=1.414)
            weight.requires_grad_(train_heads)
            attn.requires_grad_(train_heads)
            self.head_weights.append(weight)
            self.head_attn.append(attn)
        self.linear = nn.Linear(head_n, 1, device=param_device)

        # regularization
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=0)

        # sparse matrix
        self.a_sparse = a_sparse
        self.edges = a_sparse.indices()
        self.values = a_sparse.values()
        self.N = a_sparse.shape[0]
        a_dense = a_sparse.to_dense()
        a_dense[torch.where(a_dense == 0)] = -1000000000
        a_dense[torch.where(a_dense == 1)] = 0
        self.mask = a_dense

    def forward(self, x):
        b, n, s = x.shape
        x = x.reshape(b*n, s)

        atts_stack = []
        # multi-heads attention
        for n in range(self.head_n):
            h = torch.matmul(x, self.head_weights[n])   # [b*n, out_dim]
            edge_h = torch.cat((h[self.edges[0, :], :], h[self.edges[1, :], :]), dim=1).t()  # [2 * out_dim, num_edges]
            atts = self.head_attn[n].mm(edge_h).squeeze()  # [num_edges]
            atts = self.leakyrelu(atts)
            atts_stack.append(atts)

        mt_atts = torch.stack(atts_stack, dim=1) # [num_edges, head_n]
        mt_atts = self.linear(mt_atts) # [num_edges, 1]
        new_values = self.values * mt_atts.squeeze() # [num_edges]
        atts_mat = torch.sparse_coo_tensor(self.edges, new_values) # [num_edges, num_edges]
        atts_mat = atts_mat.to_dense() + self.mask # [num_edges, num_edges]
        atts_mat = self.softmax(atts_mat) # [num_edges, num_edges]
        return atts_mat

class MultiHeadsGATLayerS0(nn.Module):
    def __init__(self, a_sparse, input_dim, out_dim, head_n, dropout, alpha):
        super().__init__()
        self.head_n = head_n

        # 1) 参数注册
        self.W = nn.ParameterDict()
        self.a = nn.ParameterDict()
        for i in range(head_n):
            self.W[str(i)] = nn.Parameter(torch.empty(input_dim, out_dim, device=device))
            self.a[str(i)] = nn.Parameter(torch.empty(1, 2*out_dim, device=device))
            nn.init.xavier_normal_(self.W[str(i)], gain=1.414)
            nn.init.xavier_normal_(self.a[str(i)], gain=1.414)
        self.head_mixer = nn.Linear(head_n, 1, device=device)

        self.leakyrelu = nn.LeakyReLU(negative_slope=alpha)
        self.dropout = nn.Dropout(dropout)

        # 2) buffers
        self.N = a_sparse.shape[0]
        self.register_buffer("edges", a_sparse.indices())
        self.register_buffer("edge_values", a_sparse.values())
        adj_dense = a_sparse.to_dense()
        mask = torch.full_like(adj_dense, float('-inf'))
        mask[adj_dense > 0] = 0.0
        self.register_buffer("mask", mask)

    def forward(self, x):           # x: [b, N, S]
        b, n, s = x.shape
        x = x.reshape(b*n, s)

        atts_stack = []
        for i in range(self.head_n):
            h = x @ self.W[str(i)]                                 # [b*N, d]
            edge_h = torch.cat((h[self.edges[0]], h[self.edges[1]]), dim=1).t()  # [2d, E]
            e = (self.a[str(i)] @ edge_h).squeeze()                # [E]
            e = self.leakyrelu(e)
            atts_stack.append(e)

        mt = torch.stack(atts_stack, dim=1)                        # [E, head_n]
        mt = self.head_mixer(mt).squeeze(-1)                       # [E]
        mt = torch.clamp(mt, -8.0, 8.0)                            # 稳定
        new_values = self.edge_values * mt                         # [E]

        A = torch.sparse_coo_tensor(self.edges, new_values,
                                    size=(self.N, self.N),
                                    device=x.device, dtype=x.dtype).coalesce()
        logits = A.to_dense() + self.mask                           # [N,N]
        logits = logits - logits.max(dim=-1, keepdim=True)[0]
        attn = torch.softmax(logits, dim=-1)                        # 行归一
        attn = F.dropout(attn, p=self.dropout.p, training=self.training)

        return attn.unsqueeze(0).expand(b, -1, -1)                  # [b,N,N]

class MultiHeadsGATv2Layer(nn.Module):
    def __init__(
        self,
        a_sparse: torch.Tensor,
        input_dim: int,
        out_dim: int,
        head_n: int,
        dropout: float,
        alpha: float,
        edge_bias: torch.Tensor = None,
        per_head_beta: bool = False,
        use_linear_fuse: bool = False,   # True: 线性融合；False: 平均融合
        device=None,
        dtype=None,
    ):
        """
        a_sparse:   稀疏COO邻接（N,N），非零处为边
        input_dim:  节点输入特征维度（你这里等于 seq 长度）
        out_dim:    头内输出维度
        head_n:     多头数
        dropout:    投影上的dropout
        alpha:      LeakyReLU斜率
        edge_bias:  [E] 可选边偏置（如 -dist/sigma）；不提供则不加
        per_head_beta: True→每头一个beta；False→共享一个beta
        use_linear_fuse: True→用 learnable 线性融合多头；False→简单平均
        """
        super().__init__()
        factory_kwargs = {"device": device if device is not None else a_sparse.device,
                          "dtype": dtype}

        self.N = a_sparse.shape[0]
        self.head_n = head_n
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.use_linear_fuse = use_linear_fuse

        # --- graph structure ---
        # 注意：保存COO索引，后续统一用它构造稀疏/做edge-softmax
        a_sparse = a_sparse.coalesce()
        self.register_buffer("edges", a_sparse.indices(), persistent=False)  # [2, E]
        self.E = self.edges.shape[1]
        # 不再需要 dense mask；用 edge-softmax 按出边归一化

        # --- per-head params ---
        self.Wq = nn.ParameterList([
            nn.Parameter(torch.empty(input_dim, out_dim, **factory_kwargs))
            for _ in range(head_n)
        ])
        self.Wk = nn.ParameterList([
            nn.Parameter(torch.empty(input_dim, out_dim, **factory_kwargs))
            for _ in range(head_n)
        ])
        self.a_vec = nn.ParameterList([
            nn.Parameter(torch.empty(out_dim, 1, **factory_kwargs))
            for _ in range(head_n)
        ])

        for i in range(head_n):
            nn.init.xavier_normal_(self.Wq[i], gain=1.414)
            nn.init.xavier_normal_(self.Wk[i], gain=1.414)
            nn.init.xavier_normal_(self.a_vec[i], gain=1.414)

        # head 融合（可选）
        if use_linear_fuse:
            self.head_fuse = nn.Linear(head_n, 1, bias=True, **factory_kwargs)

        # regularization & act
        self.drop = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)

        # edge bias
        if edge_bias is not None:
            assert edge_bias.numel() == self.E, "edge_bias length must equal num_edges"
            self.register_buffer("edge_bias", edge_bias.detach().to(factory_kwargs["device"]), persistent=False)
            if per_head_beta:
                self.beta = nn.Parameter(torch.zeros(head_n, **factory_kwargs))
            else:
                self.beta = nn.Parameter(torch.zeros(1, **factory_kwargs))
        else:
            self.edge_bias = None
            self.beta = None

    @staticmethod
    def _edge_softmax(scores_e: torch.Tensor, src_idx: torch.Tensor, n_nodes: int) -> torch.Tensor:
        """
        对每个源节点src的出边scores做softmax归一化。
        scores_e: [E]
        src_idx:  [E]  (edges[0])
        return:   [E]  归一化权重
        """
        # 数值稳定的 log-sum-exp
        # 先按src分组取最大
        e_max = torch.zeros(n_nodes, device=scores_e.device, dtype=scores_e.dtype)\
                    .index_reduce_(0, src_idx, scores_e, reduce='amax')
        e_shift = scores_e - e_max.index_select(0, src_idx)
        exp_e = torch.exp(e_shift)
        denom = torch.zeros(n_nodes, device=scores_e.device, dtype=scores_e.dtype)\
                    .index_reduce_(0, src_idx, exp_e, reduce='mean')
        alpha = exp_e / (denom.index_select(0, src_idx) + 1e-12)
        return alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [b, N, S]  （S=input_dim）
        return: [b, N, N]  每个样本一张行归一的注意力矩阵
        说明：本层返回的是注意力A，后续再与特征做 bmm。
        """
        b, n, s = x.shape
        assert n == self.N and s == self.input_dim, "Input shape mismatch"

        # 合并batch与node做线性投影
        h = x.reshape(b * n, s)  # [B*N, S]
        src_idx, dst_idx = self.edges[0], self.edges[1]  # [E]

        A_heads = []
        for i in range(self.head_n):
            # 线性投影 + dropout
            Hq = self.drop(h @ self.Wq[i])   # [B*N, d]
            Hk = self.drop(h @ self.Wk[i])   # [B*N, d]

            # 取边两端
            q_i = Hq[src_idx, :]             # [E, d]
            k_j = Hk[dst_idx, :]             # [E, d]

            # GATv2打分 + 1/sqrt(d) 缩放
            e_ij = self.leakyrelu(q_i + k_j) @ self.a_vec[i]  # [E, 1]
            e_ij = e_ij.squeeze(-1) / math.sqrt(self.out_dim) # [E]

            # 可选：加边偏置
            if self.edge_bias is not None:
                if self.beta.numel() == self.head_n:
                    e_ij = e_ij + self.beta[i] * self.edge_bias
                else:
                    e_ij = e_ij + self.beta * self.edge_bias

            # 按源节点做 edge-softmax
            alpha_e = self._edge_softmax(e_ij, src_idx, self.N)  # [E]

            # 构造该头的稀疏注意力并转dense（或保留稀疏，视下游实现）
            A_head = torch.sparse_coo_tensor(
                self.edges, alpha_e, size=(self.N, self.N),
                device=h.device, dtype=h.dtype
            ).coalesce().to_dense()  # [N, N] 行归一

            A_heads.append(A_head)

        # 多头融合：平均或线性融合
        A_stack = torch.stack(A_heads, dim=-1)  # [N, N, H]
        if self.use_linear_fuse:
            # 线性融合等价于对 head 维做 Linear(H→1)
            A_fused = self.head_fuse(A_stack).squeeze(-1)  # [N, N]
        else:
            A_fused = A_stack.mean(dim=-1)                 # [N, N]

        # 扩展到 batch
        A_b = A_fused.unsqueeze(0).expand(b, -1, -1)       # [b, N, N]
        return A_b

class MultiHeadsGATv2LayerDyn(nn.Module):
    def __init__(
        self,
        a_sparse: torch.Tensor,
        input_dim: int,
        out_dim: int,
        head_n: int,
        dropout: float,
        alpha: float,
        edge_bias: torch.Tensor = None,
        per_head_beta: bool = False,
        use_linear_fuse: bool = False,   # True: 线性融合多头；False: 平均融合
        # ---- S2 动态图相关超参 ----
        dyn_sim: str = "cos",            # ["cos", "dot"]
        dyn_topk: int = 16,              # 0/None 表示不裁剪
        dyn_tau: float = 1.0,            # 温度，<1更尖锐，>1更平滑
        lambda_init: float = 0.5,        # 初始融合权重
        lambda_learnable: bool = False,  # 是否让 λ 可学习
        mask_dynamic_with_static: bool = True,  # 是否用静态边约束动态图
        device=None,
        dtype=None,
    ):
        """
        本层输出每个样本的一张注意力矩阵 A ∈ [b, N, N]（行归一）。
        - 静态分支：沿用 GATv2（支持边偏置）在静态边上做 edge-softmax 得到 A_static
        - 动态分支：由 x 构建 A_dyn（特征相似→softmax/Top-K/温度）
        - 融合：A = λ·A_static + (1-λ)·A_dyn（两者均做行归一）
        """
        super().__init__()
        factory_kwargs = {"device": device if device is not None else a_sparse.device,
                          "dtype": dtype}

        self.N = a_sparse.shape[0]
        self.head_n = head_n
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.use_linear_fuse = use_linear_fuse

        # ---- graph structure ----
        a_sparse = a_sparse.coalesce()
        self.register_buffer("edges", a_sparse.indices(), persistent=False)  # [2, E]
        self.E = self.edges.shape[1]
        self.register_buffer("A_static_mask_dense",
                             (a_sparse.to_dense() > 0).to(factory_kwargs["device"], torch.float32),
                             persistent=False)  # [N, N]，用于可选遮罩

        # ---- per-head params ----
        self.Wq = nn.ParameterList([
            nn.Parameter(torch.empty(input_dim, out_dim, **factory_kwargs))
            for _ in range(head_n)
        ])
        self.Wk = nn.ParameterList([
            nn.Parameter(torch.empty(input_dim, out_dim, **factory_kwargs))
            for _ in range(head_n)
        ])
        self.a_vec = nn.ParameterList([
            nn.Parameter(torch.empty(out_dim, 1, **factory_kwargs))
            for _ in range(head_n)
        ])
        for i in range(head_n):
            nn.init.xavier_normal_(self.Wq[i], gain=1.414)
            nn.init.xavier_normal_(self.Wk[i], gain=1.414)
            nn.init.xavier_normal_(self.a_vec[i], gain=1.414)

        if use_linear_fuse:
            self.head_fuse = nn.Linear(head_n, 1, bias=True, **factory_kwargs)

        # regularization & act
        self.drop = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)

        # edge bias
        if edge_bias is not None:
            assert edge_bias.numel() == self.E, "edge_bias length must equal num_edges"
            self.register_buffer("edge_bias", edge_bias.detach().to(factory_kwargs["device"]), persistent=False)
            if per_head_beta:
                self.beta = nn.Parameter(torch.zeros(head_n, **factory_kwargs))
            else:
                self.beta = nn.Parameter(torch.zeros(1, **factory_kwargs))
        else:
            self.edge_bias = None
            self.beta = None

        # ---- S2 动态图超参/参数 ----
        self.dyn_sim = dyn_sim
        self.dyn_topk = 0 if dyn_topk is None else int(dyn_topk)
        self.dyn_tau = float(dyn_tau)
        self.mask_dynamic_with_static = bool(mask_dynamic_with_static)

        lam = torch.tensor(float(lambda_init), **factory_kwargs)
        self.lambda_param = nn.Parameter(lam) if lambda_learnable else lam
        self.lambda_learnable = lambda_learnable

    @staticmethod
    def _edge_softmax(scores_e: torch.Tensor, src_idx: torch.Tensor, n_nodes: int) -> torch.Tensor:
        # 对每个源节点的出边做 softmax（稳定实现）
        e_max = torch.zeros(n_nodes, device=scores_e.device, dtype=scores_e.dtype)\
                    .index_reduce_(0, src_idx, scores_e, reduce='amax')
        e_shift = scores_e - e_max.index_select(0, src_idx)
        exp_e = torch.exp(e_shift)
        denom = torch.zeros(n_nodes, device=scores_e.device, dtype=scores_e.dtype)
        denom.index_add_(0, src_idx, exp_e)
        alpha = exp_e / (denom.index_select(0, src_idx) + 1e-12)
        return alpha

    @staticmethod
    def _row_normalize_dense(A: torch.Tensor) -> torch.Tensor:
        return A / (A.sum(dim=-1, keepdim=True) + 1e-12)

    def _build_A_static(self, x: torch.Tensor) -> torch.Tensor:
        """
        基于静态边的 GATv2 注意力（含可选边偏置）→ dense 行归一 A_static ∈ [N, N]
        注意：对 batch 内所有样本共享一张静态图（与原实现一致）
        """
        b, n, s = x.shape
        h = x.reshape(b * n, s)  # [B*N, S]
        src_idx, dst_idx = self.edges[0], self.edges[1]

        A_heads = []
        for i in range(self.head_n):
            Hq = self.drop(h @ self.Wq[i])   # [B*N, d]
            Hk = self.drop(h @ self.Wk[i])   # [B*N, d]
            q_i = Hq[src_idx, :]             # [E, d]
            k_j = Hk[dst_idx, :]             # [E, d]

            e_ij = self.leakyrelu(q_i + k_j) @ self.a_vec[i]    # [E, 1]
            e_ij = e_ij.squeeze(-1) / math.sqrt(self.out_dim)   # [E]

            if self.edge_bias is not None:
                if isinstance(self.beta, nn.Parameter) and self.beta.numel() == self.head_n:
                    e_ij = e_ij + self.beta[i] * self.edge_bias
                elif self.beta is not None:
                    e_ij = e_ij + self.beta * self.edge_bias

            alpha_e = self._edge_softmax(e_ij, src_idx, self.N)  # [E]

            A_head = torch.sparse_coo_tensor(
                self.edges, alpha_e, size=(self.N, self.N),
                device=h.device, dtype=h.dtype
            ).coalesce().to_dense()  # [N, N] 行归一
            A_heads.append(A_head)

        A_stack = torch.stack(A_heads, dim=-1)  # [N, N, H]
        if self.use_linear_fuse:
            A_static = self.head_fuse(A_stack).squeeze(-1)  # [N, N]
        else:
            A_static = A_stack.mean(dim=-1)                 # [N, N]

        # 切实保证行归一（多头融合后再归一一次更稳）
        A_static = self._row_normalize_dense(A_static)
        return A_static  # [N, N]

    def _build_A_dynamic(self, x: torch.Tensor) -> torch.Tensor:
        """
        由本层输入 x 构建动态相似度图（dense）并做行归一
        - 相似度：cos / dot
        - 支持 Top-K 裁剪
        - 可选用静态边做遮罩
        """
        # 用最后一维作为特征：先做一个简单投影可提升稳健（也可直接用 x）
        # 这里直接用 x 的最后一维特征聚合：mean over seq，等价于 avg-pool
        # 你也可以换成 nn.Linear(input_dim, d_dyn) 做可学习投影
        with torch.no_grad():
            pass
        # x: [b, N, S] → 节点级表示 h_dyn: [b, N, S]
        h_dyn = x  # 若要线性投影，可在 __init__ 加 self.proj_dyn

        if self.dyn_sim == "cos":
            # 归一化后做余弦
            h = F.normalize(h_dyn, p=2, dim=-1)      # [b, N, S]
            A = torch.matmul(h, h.transpose(1, 2))   # [b, N, N] ∈ [-1,1]
        elif self.dyn_sim == "dot":
            A = torch.matmul(h_dyn, h_dyn.transpose(1, 2))  # [b, N, N]
        else:
            raise ValueError("dyn_sim must be 'cos' or 'dot'.")

        # 温度缩放（放在 softmax 前）
        A = A / max(self.dyn_tau, 1e-6)

        # 可选：用静态边约束动态图（仅保留静态边）
        if self.mask_dynamic_with_static:
            A = A.masked_fill(self.A_static_mask_dense.unsqueeze(0) == 0, float("-inf"))
        
        # Top-K 稀疏化（对每行）
        if self.dyn_topk and self.dyn_topk > 0 and self.dyn_topk < self.N:
            topk_val, topk_idx = torch.topk(A, k=self.dyn_topk, dim=-1)
            mask = torch.zeros_like(A)
            mask.scatter_(-1, topk_idx, 1.0)
            A = A.masked_fill(mask == 0, float("-inf"))

        # 行 softmax → 概率型传递矩阵
        A_dyn = F.softmax(A, dim=-1)  # [b, N, N]
        return A_dyn

    def forward(self, x: torch.Tensor, use_dynamic: bool = False) -> torch.Tensor:
        """
        x: [b, N, S]
        return: [b, N, N]  行归一的融合注意力
        - 第一层：use_dynamic=False（保持稳定）
        - 第二层：use_dynamic=True（启用 S2）
        """
        b, n, s = x.shape
        assert n == self.N and s == self.input_dim, "Input shape mismatch"

        # 静态注意力（共享一张图，expand 到 batch）
        A_static = self._build_A_static(x)                 # [N, N]
        A_static = self._row_normalize_dense(A_static)     # 稳一遍
        A_static_b = A_static.unsqueeze(0).expand(b, -1, -1)

        if not use_dynamic:
            return A_static_b  # S1-only

        # 动态图
        A_dyn = self._build_A_dynamic(x)              # [b, N, N]
        A_dyn = self._row_normalize_dense(A_dyn)

        # λ 融合（若 learnable 则做 sigmoid 限制到 0~1）
        lam = torch.sigmoid(self.lambda_param) if self.lambda_learnable else torch.clamp(self.lambda_param, 0.0, 1.0)
        A_fused = lam * A_static_b + (1.0 - lam) * A_dyn   # [b, N, N]
        # 最后再行归一一次，确保数值稳定
        A_fused = self._row_normalize_dense(A_fused)
        return A_fused

class GatedTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels * 2,
                              kernel_size, padding=padding, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.residual.weight, nonlinearity='linear')

    def forward(self, x):
        """
        x: [B, C_in, T]
        """
        out = self.conv(x)[:, :, :-self.conv.dilation[0]*(self.conv.kernel_size[0]-1)]  # 保证因果
        A, B = out.chunk(2, dim=1)
        out = torch.tanh(A) * torch.sigmoid(B)
        out = self.dropout(out)
        res = self.residual(x)
        return out + res

class TemporalConvNet(nn.Module):
    def __init__(self, in_channels=2, hidden_channels=32, n_layers=3,
                 kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        for i in range(n_layers):
            dilation = 2 ** i
            layers.append(GatedTCNBlock(
                in_channels if i == 0 else hidden_channels,
                hidden_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout
            ))
        self.network = nn.Sequential(*layers)
        self.proj = nn.Conv1d(hidden_channels, 1, 1)

    def forward(self, x):
        """
        x: [B, N, T, C]  →  [B*N, C, T]
        return: [B, N, T']
        """
        B, N, T, C = x.shape
        x = x.reshape(B * N, C, T)
        y = self.network(x)
        y = self.proj(y)  # [B*N, 1, T]
        y = y.squeeze(1).reshape(B, N, -1)
        return y

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def causal_mask(T, device):
        # shape [T, T], True 表示“被遮住”（不可见）
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x):
        """
        x: [B*, S, d_model]  （这里 B* = B*N）
        """
        Bstar, S, d = x.shape
        mask = self.causal_mask(S, x.device)
        # MHA 残差
        h = self.ln1(x)
        attn_out, _ = self.mha(h, h, h, attn_mask=mask)
        x = x + self.drop(attn_out)
        # FFN 残差
        h = self.ln2(x)
        x = x + self.drop(self.ffn(h))
        return x  # [B*, S, d_model]


class CausalTransformerEncoder(nn.Module):
    def __init__(self, d_model=128, n_heads=4, depth=2, d_ff=256, dropout=0.1, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))  # 可学习位置编码
        nn.init.normal_(self.pos_emb, std=0.02)
        self.blocks = nn.ModuleList([
            CausalSelfAttention(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(depth)
        ])
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: [B*, S, d_model]
        """
        Bstar, S, d = x.shape
        x = x + self.pos_emb[:, :S, :]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return x  # [B*, S, d_model]

class TemporalHeadTransformer(nn.Module):
    """
    输入：来自 GAT 的时序特征，形状 [B, N, S]
    处理：1→d_model 线性投影 + 因果 Transformer
    输出：每节点标量预测 [B, N]
    """
    def __init__(self, d_model=128, n_heads=4, depth=2, d_ff=256, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(1, d_model)
        self.encoder = CausalTransformerEncoder(d_model=d_model, n_heads=n_heads,
                                                depth=depth, d_ff=d_ff, dropout=dropout, max_len=1024)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(self, x_seq):
        """
        x_seq: [B, N, S]
        """
        B, N, S = x_seq.shape
        x = x_seq.unsqueeze(-1)                    # [B, N, S, 1]
        x = x.view(B*N, S, 1)                      # [B*N, S, 1]
        x = self.in_proj(x)                        # [B*N, S, d_model]
        x = self.encoder(x)                        # [B*N, S, d_model]
        h_last = x[:, -1, :]                       # [B*N, d_model]
        y = self.head(h_last).view(B, N)           # [B, N]
        return y


class PAG(nn.Module):
    def __init__(self, a_sparse, seq=12, kcnn=2, k=6, m=2):
        super(PAG, self).__init__()
        self.feature = seq
        self.seq = seq-kcnn+1
        self.alpha = 0.5
        self.m = m
        self.a_sparse = a_sparse
        self.nodes = a_sparse.shape[0]

        # GAT
        self.conv2d = nn.Conv2d(1, 1, (kcnn, 2))  # input.shape = [batch, channel, width, height]
        self.gat_lyr = MultiHeadsGATLayer(a_sparse, self.seq, self.seq, 4, 0, 0.2)
        self.gcn = nn.Linear(in_features=self.seq, out_features=self.seq)

        # TPA
        self.lstm = nn.LSTM(m, m, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(in_features=self.seq - 1, out_features=k)
        self.fc2 = nn.Linear(in_features=k, out_features=m)
        self.fc3 = nn.Linear(in_features=k + m, out_features=1)
        self.decoder = nn.Linear(self.seq, 1)

        # Activation
        self.dropout = nn.Dropout(p=0.5)
        self.LeakyReLU = nn.LeakyReLU()

        #
        adj1 = copy.deepcopy(self.a_sparse.to_dense())
        adj2 = copy.deepcopy(self.a_sparse.to_dense())
        for i in range(self.nodes):
            adj1[i, i] = 0.000000001
            adj2[i, i] = 0
        degree = 1.0 / (torch.sum(adj1, dim=0))
        degree_matrix = torch.zeros((self.nodes, self.feature), device=device)
        for i in range(12):
            degree_matrix[:, i] = degree
        self.degree_matrix = degree_matrix
        self.adj2 = adj2

    def forward(self, occ, prc):  # occ.shape = [batch,node, seq]
        b, n, s = occ.shape
        data = torch.stack([occ, prc], dim=3).reshape(b*n, s, -1).unsqueeze(1)
        print("conv2d input mean/std:", data.mean().item(), data.std().item())
        data = self.conv2d(data)
        print("conv2d output mean/std:", data.mean().item(), data.std().item())
        data = data.squeeze().reshape(b, n, -1)

        # first layer
        atts_mat = self.gat_lyr(data)  # attention matrix, dense(nodes, nodes)
        occ_conv1 = torch.matmul(atts_mat, data)  # (b, n, s)
        occ_conv1 = self.dropout(self.LeakyReLU(self.gcn(occ_conv1)))

        # second layer
        atts_mat2 = self.gat_lyr(occ_conv1)  # attention matrix, dense(nodes, nodes)
        occ_conv2 = torch.matmul(atts_mat2, occ_conv1)  # (b, n, s)
        occ_conv2 = self.dropout(self.LeakyReLU(self.gcn(occ_conv2)))

        occ_conv1 = (1 - self.alpha) * occ_conv1 + self.alpha * data
        occ_conv2 = (1 - self.alpha) * occ_conv2 + self.alpha * occ_conv1
        occ_conv1 = occ_conv1.view(b * n, self.seq)
        occ_conv2 = occ_conv2.view(b * n, self.seq)

        x = torch.stack([occ_conv1, occ_conv2], dim=2)  # best
        lstm_out, (_, _) = self.lstm(x)  # b*n, s, 2

        # TPA
        ht = lstm_out[:, -1, :]  # ht [b*n, 2]
        hw = lstm_out[:, :-1, :]  # from h(t-1) to h1 [b*n, s-1, 2]
        hw = torch.transpose(hw, 1, 2) # [b*n, 2, s-1]
        Hc = self.fc1(hw) # [b*n, 2, k]
        Hn = self.fc2(Hc) # [b*n, 2, m]
        ht = torch.unsqueeze(ht, dim=2) # [b*n, 2, 1]
        a = torch.bmm(Hn, ht) # [b*n, 2, 1]
        a = torch.sigmoid(a)  # [b*n, 2, 1]
        a = torch.transpose(a, 1, 2) # [b*n, 1, 2]
        vt = torch.matmul(a, Hc) # [b*n, 1, k]
        ht = torch.transpose(ht, 1, 2) # [b*n, 1, 2]
        hx = torch.cat((vt, ht), dim=2) # [b*n, 1, k+2]
        y = self.fc3(hx) # [b*n, 1]
        y = y.view(b, n)
        return y

class PAGS0(nn.Module):
    def __init__(self, a_sparse, seq=12, kcnn=2, k=6, m=2):
        super(PAGS0, self).__init__()
        self.feature = seq
        self.seq = seq-kcnn+1
        self.alpha = 0.5
        self.m = m
        self.a_sparse = a_sparse
        self.nodes = a_sparse.shape[0]

        # GAT
        self.conv2d = nn.Conv2d(1, 1, (kcnn, 2))  # input.shape = [batch, channel, width, height]
        self.gat_lyr = MultiHeadsGATLayerS0(a_sparse, self.seq, self.seq, 4, 0, 0.2)
        self.gcn = nn.Linear(in_features=self.seq, out_features=self.seq)

        # TPA
        self.lstm = nn.LSTM(m, m, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(in_features=self.seq - 1, out_features=k)
        self.fc2 = nn.Linear(in_features=k, out_features=m)
        self.fc3 = nn.Linear(in_features=k + m, out_features=1)
        self.decoder = nn.Linear(self.seq, 1)

        # Activation
        self.dropout = nn.Dropout(p=0.5)
        self.LeakyReLU = nn.LeakyReLU()

        #
        adj1 = copy.deepcopy(self.a_sparse.to_dense())
        adj2 = copy.deepcopy(self.a_sparse.to_dense())
        for i in range(self.nodes):
            adj1[i, i] = 0.000000001
            adj2[i, i] = 0
        degree = 1.0 / (torch.sum(adj1, dim=0))
        degree_matrix = torch.zeros((self.nodes, self.feature), device=device)
        for i in range(12):
            degree_matrix[:, i] = degree
        self.degree_matrix = degree_matrix
        self.adj2 = adj2

    def forward(self, occ, prc):  # occ.shape = [batch,node, seq]
        b, n, s = occ.shape
        data = torch.stack([occ, prc], dim=3).reshape(b*n, s, -1).unsqueeze(1)
        data = self.conv2d(data)
        data = data.squeeze().reshape(b, n, -1)

        # first layer
        atts_mat = self.gat_lyr(data)  # attention matrix, dense(nodes, nodes)
        occ_conv1 = torch.bmm(atts_mat, data)  # (b, n, s)
        occ_conv1 = self.dropout(self.LeakyReLU(self.gcn(occ_conv1)))

        # second layer
        atts_mat2 = self.gat_lyr(occ_conv1)  # attention matrix, dense(nodes, nodes)
        occ_conv2 = torch.bmm(atts_mat2, occ_conv1)  # (b, n, s)
        occ_conv2 = self.dropout(self.LeakyReLU(self.gcn(occ_conv2)))

        occ_conv1 = (1 - self.alpha) * occ_conv1 + self.alpha * data
        occ_conv2 = (1 - self.alpha) * occ_conv2 + self.alpha * occ_conv1
        occ_conv1 = occ_conv1.view(b * n, self.seq)
        occ_conv2 = occ_conv2.view(b * n, self.seq)

        x = torch.stack([occ_conv1, occ_conv2], dim=2)  # best
        lstm_out, (_, _) = self.lstm(x)  # b*n, s, 2

        # TPA
        ht = lstm_out[:, -1, :]  # ht [b*n, 2]
        hw = lstm_out[:, :-1, :]  # from h(t-1) to h1 [b*n, s-1, 2]
        hw = torch.transpose(hw, 1, 2) # [b*n, 2, s-1]
        Hc = self.fc1(hw) # [b*n, 2, k]
        Hn = self.fc2(Hc) # [b*n, 2, m]
        ht = torch.unsqueeze(ht, dim=2) # [b*n, 2, 1]
        a = torch.bmm(Hn, ht) # [b*n, 2, 1]
        a = torch.sigmoid(a)  # [b*n, 2, 1]
        a = torch.transpose(a, 1, 2) # [b*n, 1, 2]
        vt = torch.matmul(a, Hc) # [b*n, 1, k]
        ht = torch.transpose(ht, 1, 2) # [b*n, 1, 2]
        hx = torch.cat((vt, ht), dim=2) # [b*n, 1, k+2]
        y = self.fc3(hx) # [b*n, 1]
        y = y.view(b, n)
        return y

class PAGS1(nn.Module):
    def __init__(self, a_sparse, seq=12, kcnn=2, k=6, m=2):
        super(PAGS1, self).__init__()
        self.feature = seq
        self.seq = seq-kcnn+1
        self.alpha = 0.5
        self.m = m
        self.a_sparse = a_sparse
        self.nodes = a_sparse.shape[0]

        # GAT
        self.conv2d = nn.Conv2d(1, 1, (kcnn, 2))  # input.shape = [batch, channel, width, height]
        self.gat_lyr = MultiHeadsGATv2Layer(a_sparse, self.seq, self.seq, 4, 0.1, 0.2, edge_bias=None, device=device)
        self.gcn = nn.Linear(in_features=self.seq, out_features=self.seq)

        # TPA
        self.lstm = nn.LSTM(m, m, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(in_features=self.seq - 1, out_features=k)
        self.fc2 = nn.Linear(in_features=k, out_features=m)
        self.fc3 = nn.Linear(in_features=k + m, out_features=1)
        self.decoder = nn.Linear(self.seq, 1)

        # Activation
        self.dropout = nn.Dropout(p=0.5)
        self.LeakyReLU = nn.LeakyReLU()

        #
        adj1 = copy.deepcopy(self.a_sparse.to_dense())
        adj2 = copy.deepcopy(self.a_sparse.to_dense())
        for i in range(self.nodes):
            adj1[i, i] = 0.000000001
            adj2[i, i] = 0
        degree = 1.0 / (torch.sum(adj1, dim=0))
        degree_matrix = torch.zeros((self.nodes, self.feature), device=device)
        for i in range(12):
            degree_matrix[:, i] = degree
        self.degree_matrix = degree_matrix
        self.adj2 = adj2

    def forward(self, occ, prc):  # occ.shape = [batch,node, seq]
        b, n, s = occ.shape
        data = torch.stack([occ, prc], dim=3).reshape(b*n, s, -1).unsqueeze(1)
        data = self.conv2d(data)
        data = data.squeeze().reshape(b, n, -1)

        # first layer
        atts_mat = self.gat_lyr(data)  # attention matrix, dense(nodes, nodes)
        occ_conv1 = torch.bmm(atts_mat, data)  # (b, n, s)
        occ_conv1 = self.dropout(self.LeakyReLU(self.gcn(occ_conv1)))

        # second layer
        atts_mat2 = self.gat_lyr(occ_conv1)  # attention matrix, dense(nodes, nodes)
        occ_conv2 = torch.bmm(atts_mat2, occ_conv1)  # (b, n, s)
        occ_conv2 = self.dropout(self.LeakyReLU(self.gcn(occ_conv2)))

        occ_conv1 = (1 - self.alpha) * occ_conv1 + self.alpha * data
        occ_conv2 = (1 - self.alpha) * occ_conv2 + self.alpha * occ_conv1
        occ_conv1 = occ_conv1.view(b * n, self.seq)
        occ_conv2 = occ_conv2.view(b * n, self.seq)

        x = torch.stack([occ_conv1, occ_conv2], dim=2)  # best
        lstm_out, (_, _) = self.lstm(x)  # b*n, s, 2

        # TPA
        ht = lstm_out[:, -1, :]  # ht [b*n, 2]
        hw = lstm_out[:, :-1, :]  # from h(t-1) to h1 [b*n, s-1, 2]
        hw = torch.transpose(hw, 1, 2) # [b*n, 2, s-1]
        Hc = self.fc1(hw) # [b*n, 2, k]
        Hn = self.fc2(Hc) # [b*n, 2, m]
        ht = torch.unsqueeze(ht, dim=2) # [b*n, 2, 1]
        a = torch.bmm(Hn, ht) # [b*n, 2, 1]
        a = torch.sigmoid(a)  # [b*n, 2, 1]
        a = torch.transpose(a, 1, 2) # [b*n, 1, 2]
        vt = torch.matmul(a, Hc) # [b*n, 1, k]
        ht = torch.transpose(ht, 1, 2) # [b*n, 1, 2]
        hx = torch.cat((vt, ht), dim=2) # [b*n, 1, k+2]
        y = self.fc3(hx) # [b*n, 1]
        y = y.view(b, n)
        return y

class PAGS2(nn.Module):
    def __init__(self, a_sparse, seq=12, kcnn=2, k=6, m=2):
        super(PAGS2, self).__init__()
        self.feature = seq
        self.seq = seq-kcnn+1
        self.alpha = 0.5
        self.m = m
        self.a_sparse = a_sparse
        self.nodes = a_sparse.shape[0]

        # GAT
        self.conv2d = nn.Conv2d(1, 1, (kcnn, 2))  # input.shape = [batch, channel, width, height]
        self.gat_lyr = MultiHeadsGATv2LayerDyn(a_sparse, self.seq, self.seq, 4, 0.1, 0.2, edge_bias=None, device=device)
        self.gcn = nn.Linear(in_features=self.seq, out_features=self.seq)

        # TPA
        self.lstm = nn.LSTM(m, m, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(in_features=self.seq - 1, out_features=k)
        self.fc2 = nn.Linear(in_features=k, out_features=m)
        self.fc3 = nn.Linear(in_features=k + m, out_features=1)
        self.decoder = nn.Linear(self.seq, 1)

        # Activation
        self.dropout = nn.Dropout(p=0.5)
        self.LeakyReLU = nn.LeakyReLU()

        #
        adj1 = copy.deepcopy(self.a_sparse.to_dense())
        adj2 = copy.deepcopy(self.a_sparse.to_dense())
        for i in range(self.nodes):
            adj1[i, i] = 0.000000001
            adj2[i, i] = 0
        degree = 1.0 / (torch.sum(adj1, dim=0))
        degree_matrix = torch.zeros((self.nodes, self.feature), device=device)
        for i in range(12):
            degree_matrix[:, i] = degree
        self.degree_matrix = degree_matrix
        self.adj2 = adj2

    def forward(self, occ, prc):  # occ.shape = [batch,node, seq]
        b, n, s = occ.shape
        data = torch.stack([occ, prc], dim=3).reshape(b*n, s, -1).unsqueeze(1)
        data = self.conv2d(data)
        data = data.squeeze().reshape(b, n, -1)

        # first layer
        atts_mat = self.gat_lyr(data, use_dynamic=False)  # attention matrix, dense(nodes, nodes)
        occ_conv1 = torch.bmm(atts_mat, data)  # (b, n, s)
        occ_conv1 = self.dropout(self.LeakyReLU(self.gcn(occ_conv1)))

        # second layer
        atts_mat2 = self.gat_lyr(occ_conv1, use_dynamic=True)  # attention matrix, dense(nodes, nodes)
        occ_conv2 = torch.bmm(atts_mat2, occ_conv1)  # (b, n, s)
        occ_conv2 = self.dropout(self.LeakyReLU(self.gcn(occ_conv2)))

        occ_conv1 = (1 - self.alpha) * occ_conv1 + self.alpha * data
        occ_conv2 = (1 - self.alpha) * occ_conv2 + self.alpha * occ_conv1
        occ_conv1 = occ_conv1.view(b * n, self.seq)
        occ_conv2 = occ_conv2.view(b * n, self.seq)

        x = torch.stack([occ_conv1, occ_conv2], dim=2)  # best
        lstm_out, (_, _) = self.lstm(x)  # b*n, s, 2

        # TPA
        ht = lstm_out[:, -1, :]  # ht [b*n, 2]
        hw = lstm_out[:, :-1, :]  # from h(t-1) to h1 [b*n, s-1, 2]
        hw = torch.transpose(hw, 1, 2) # [b*n, 2, s-1]
        Hc = self.fc1(hw) # [b*n, 2, k]
        Hn = self.fc2(Hc) # [b*n, 2, m]
        ht = torch.unsqueeze(ht, dim=2) # [b*n, 2, 1]
        a = torch.bmm(Hn, ht) # [b*n, 2, 1]
        a = torch.sigmoid(a)  # [b*n, 2, 1]
        a = torch.transpose(a, 1, 2) # [b*n, 1, 2]
        vt = torch.matmul(a, Hc) # [b*n, 1, k]
        ht = torch.transpose(ht, 1, 2) # [b*n, 1, 2]
        hx = torch.cat((vt, ht), dim=2) # [b*n, 1, k+2]
        y = self.fc3(hx) # [b*n, 1]
        y = y.view(b, n)
        return y

class PAGS3(nn.Module):
    def __init__(self, a_sparse, seq=12, k=6, m=2):
        super(PAGS3, self).__init__()
        self.seq = seq
        self.alpha = 0.5
        self.m = m
        self.a_sparse = a_sparse
        self.nodes = a_sparse.shape[0]

        # ---- S3: TemporalConvNet 取代 Conv2D ----
        self.tcn = TemporalConvNet(in_channels=2, hidden_channels=32, n_layers=3, dropout=0.1)

        # ---- 空间建模部分 (保留 S2 GATv2+动态图) ----
        self.gat_lyr = MultiHeadsGATv2LayerDyn(a_sparse, seq, seq, 4, 0.1, 0.2)
        self.gcn = nn.Linear(in_features=seq, out_features=seq)

        # ---- 时间预测部分 (原始 TPA 不动) ----
        self.lstm = nn.LSTM(m, m, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(seq - 1, k)
        self.fc2 = nn.Linear(k, m)
        self.fc3 = nn.Linear(k + m, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.LeakyReLU = nn.LeakyReLU()

    def forward(self, occ, prc):
        b, n, s = occ.shape
        x_in = torch.stack([occ, prc], dim=-1)  # [B, N, S, 2]
        print("TCN input mean/std:", x_in.mean().item(), x_in.std().item())
        data = self.tcn(x_in)                   # [B, N, S’]
        print("TCN output mean/std:", data.mean().item(), data.std().item())

        # GAT Block ×2
        atts_mat = self.gat_lyr(data, use_dynamic=False)
        occ_conv1 = torch.bmm(atts_mat, data)
        occ_conv1 = self.dropout(self.LeakyReLU(self.gcn(occ_conv1)))

        atts_mat2 = self.gat_lyr(occ_conv1, use_dynamic=True)
        occ_conv2 = torch.bmm(atts_mat2, occ_conv1)
        occ_conv2 = self.dropout(self.LeakyReLU(self.gcn(occ_conv2)))

        # 残差平滑
        occ_conv1 = (1 - self.alpha) * occ_conv1 + self.alpha * data
        occ_conv2 = (1 - self.alpha) * occ_conv2 + self.alpha * occ_conv1

        # 送入 TPA
        occ_conv1 = occ_conv1.view(b * n, self.seq)
        occ_conv2 = occ_conv2.view(b * n, self.seq)
        x = torch.stack([occ_conv1, occ_conv2], dim=2)
        lstm_out, _ = self.lstm(x)
        ht = lstm_out[:, -1, :]
        hw = torch.transpose(lstm_out[:, :-1, :], 1, 2)
        Hc = self.fc1(hw)
        Hn = self.fc2(Hc)
        ht = ht.unsqueeze(2)
        a = torch.sigmoid(torch.bmm(Hn, ht))
        a = torch.transpose(a, 1, 2)
        vt = torch.matmul(a, Hc)
        ht = ht.transpose(1, 2)
        hx = torch.cat((vt, ht), dim=2)
        y = self.fc3(hx).view(b, n)
        return y

class PAGS4(nn.Module):
    def __init__(self,
                 a_sparse,
                 seq=12,
                 tcn_hidden=32,
                 tcn_layers=3,
                 gat_heads=4,
                 gat_dropout=0.1,
                 gat_alpha=0.2,
                 tr_d_model=128,
                 tr_heads=4,
                 tr_depth=2,
                 tr_d_ff=256,
                 tr_dropout=0.1):
        super().__init__()
        self.seq = seq
        self.alpha = 0.5
        self.nodes = a_sparse.shape[0]

        # ---- S3: 短期时序特征（因果TCN） ----
        self.tcn = TemporalConvNet(in_channels=2, hidden_channels=tcn_hidden,
                                   n_layers=tcn_layers, dropout=0.1)

        # ---- S2: 空间图（GATv2 + 动态图融合）----
        self.gat_lyr = MultiHeadsGATv2LayerDyn(
            a_sparse, input_dim=seq, out_dim=seq, head_n=gat_heads,
            dropout=gat_dropout, alpha=gat_alpha,
            edge_bias=None, lambda_learnable=True  # 建议把 λ 设为可学习
        )
        self.gcn = nn.Linear(in_features=seq, out_features=seq)
        self.dropout = nn.Dropout(p=0.5)
        self.act = nn.LeakyReLU()

        # ---- S4: 因果 Transformer 时序头（替代 LSTM+TPA）----
        self.temporal_head = TemporalHeadTransformer(
            d_model=tr_d_model, n_heads=tr_heads, depth=tr_depth, d_ff=tr_d_ff, dropout=tr_dropout
        )

    def forward(self, occ, prc):
        """
        occ, prc: [B, N, S]
        return:   [B, N]
        """
        B, N, S = occ.shape
        # 1) 短期卷积
        x_in = torch.stack([occ, prc], dim=-1)   # [B, N, S, 2]
        data = self.tcn(x_in)                    # [B, N, S']（实现里已经对齐为 S 或 S'≤S）

        # 若 TCN 造成长度变化，统一裁成当前序列长度
        S_cur = data.size(-1)
        if S_cur != self.seq:
            # 与 GAT设定一致：这里按最短裁剪
            S_eff = min(S_cur, self.seq)
            data = data[:, :, -S_eff:]           # 取最近 S_eff
        else:
            S_eff = self.seq

        # 2) GAT × 2
        A1 = self.gat_lyr(data[:, :, :S_eff], use_dynamic=False)   # [B, N, N]
        h1 = torch.bmm(A1, data[:, :, :S_eff])                      # [B, N, S_eff]
        h1 = self.dropout(self.act(self.gcn(h1)))

        A2 = self.gat_lyr(h1, use_dynamic=True)
        h2 = torch.bmm(A2, h1)
        h2 = self.dropout(self.act(self.gcn(h2)))

        # 残差
        h1 = (1 - self.alpha) * h1 + self.alpha * data[:, :, :S_eff]
        h2 = (1 - self.alpha) * h2 + self.alpha * h1                # [B, N, S_eff]

        # 3) 因果 Transformer 头（替代 LSTM+TPA）
        y = self.temporal_head(h2)                                  # [B, N]
        return y

class Informer(nn.Module):
    def __init__(
        self,
        input_dim,
        seq_len,
        pred_len,
        d_model=128,
        n_heads=4,
        e_layers=4,
        d_layers=4,
        batch_first=True,
    ):
        super(Informer, self).__init__()
        self.batch_first = batch_first
        self.input_proj = nn.Linear(input_dim, d_model)  # Projection layer to transform input_dim to d_model
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=batch_first),
            num_layers=e_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, batch_first=batch_first),
            num_layers=d_layers
        )
        # self.fc_out = nn.Linear(d_model, pred_len)

    def forward(self, x):
        b, n, s, f = x.shape  # x is 4D tensor
        x = x.view(b, n, s * f)  # Merge the last two dimensions
        x = self.input_proj(x)  # Transform input dimension to d_model
        enc_out = self.encoder(x)
        dec_out = self.decoder(enc_out, enc_out)
        return dec_out

class PAGInformer(nn.Module):
    def __init__(self, a_sparse, seq=12, pred_len=6, hidden_dim=128):
        super(PAGInformer, self).__init__()
        self.seq = seq
        self.pred_len = pred_len
        self.alpha = 0.5
        self.a_sparse = a_sparse
        self.nodes = a_sparse.shape[0]
        self.gat_layer = MultiHeadsGATLayer(a_sparse, seq, seq, head_n=4, dropout=0, alpha=0.2)
        self.gcn = nn.Linear(in_features=seq, out_features=seq)
        self.informer = Informer(input_dim=seq*2, seq_len=seq, pred_len=pred_len, d_model=hidden_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, occ, prc):
        b, n, s = occ.shape
        data = torch.stack([occ, prc], dim=3).reshape(2 * b, n, s)
        atts_mat = self.gat_layer(data)
        occ_conv1 = torch.matmul(atts_mat, data)
        occ_conv1 = self.dropout(self.leakyrelu(self.gcn(occ_conv1)))
        atts_mat2 = self.gat_layer(occ_conv1)
        occ_conv2 = torch.matmul(atts_mat2, occ_conv1)
        occ_conv2 = self.dropout(self.leakyrelu(self.gcn(occ_conv2)))
        occ_conv1 = (1 - self.alpha) * occ_conv1 + self.alpha * data
        occ_conv2 = (1 - self.alpha) * occ_conv2 + self.alpha * occ_conv1
        occ_conv2 = occ_conv2.view(b, n, s, -1)
        y = self.informer(occ_conv2)

        return y[:, :, -1].view(b, n)

class PAGInformerAblation(nn.Module):
    """
    Ablation-ready version of PAGInformer.
    每个组件都可以通过参数置为 None → 实现消融。
    """
    def __init__(
        self,
        a_sparse,
        seq=12,
        pred_len=6,
        hidden_dim=128,
        use_gat=True,
        use_gcn=True,
        use_residual=True,
        use_informer=True
    ):
        super().__init__()

        self.seq = seq
        self.pred_len = pred_len
        self.alpha = 0.5
        self.nodes = a_sparse.shape[0]

        # ===== Ablation switches =====
        self.use_gat = use_gat
        self.use_gcn = use_gcn
        self.use_residual = use_residual
        self.use_informer = use_informer

        # ===== Components =====
        if use_gat:
            self.gat_layer = MultiHeadsGATLayer(a_sparse, seq, seq, head_n=4, dropout=0, alpha=0.2)

        if use_gcn:
            self.gcn = nn.Linear(in_features=seq, out_features=seq)

        if use_informer:
            self.informer = Informer(input_dim=seq*2, seq_len=seq, pred_len=pred_len, d_model=hidden_dim)
        else:
            # no informer → 用简单 MLP 替代
            self.mlp_predictor = nn.Sequential(
                nn.Linear(seq * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

        self.leakyrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)


    def _apply_gcn(self, x):
        """GCN 或 identity"""
        if self.use_gcn:
            return self.dropout(self.leakyrelu(self.gcn(x)))
        return x


    def forward(self, occ, prc):
        b, n, s = occ.shape

        # ① 输入拼接
        data = torch.stack([occ, prc], dim=3).reshape(2 * b, n, s)

        # ============ GAT & GCN Block 1 ============
        if self.use_gat:
            atts_mat = self.gat_layer(data)
            occ_conv1 = torch.matmul(atts_mat, data)
        else:
            # 不使用 GAT → identity
            occ_conv1 = data

        occ_conv1 = self._apply_gcn(occ_conv1)

        # ============ GAT & GCN Block 2 ============
        if self.use_gat:
            atts_mat2 = self.gat_layer(occ_conv1)
            occ_conv2 = torch.matmul(atts_mat2, occ_conv1)
        else:
            occ_conv2 = occ_conv1

        occ_conv2 = self._apply_gcn(occ_conv2)

        # ============ residual connection ============
        if self.use_residual:
            occ_conv1 = (1 - self.alpha) * occ_conv1 + self.alpha * data
            occ_conv2 = (1 - self.alpha) * occ_conv2 + self.alpha * occ_conv1

        # reshape for Informer
        occ_conv2 = occ_conv2.view(b, n, s, -1)

        # ============ Informer / MLP ============
        if self.use_informer:                             
            y = self.informer(occ_conv2)
            return y[:, :, -1].view(b, n)
        else:
            # 平均池化 → MLP
            pooled = occ_conv2.reshape(b, n, s * 2)
            out = self.mlp_predictor(pooled)
            return out.view(b, n)

class PAGInformerQuantile(nn.Module):
    def __init__(
        self,
        a_sparse,
        seq=12,
        pred_len=6,
        hidden_dim=128,
        quantiles=None,
        horizons=None,
        train_gat_heads=True,
        transformer_batch_first=True,
    ):
        super(PAGInformerQuantile, self).__init__()

        if quantiles is None:
            quantiles = [0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95]
        if horizons is None:
            horizons = [1]

        self.quantiles = [float(q) for q in quantiles]
        self.Q = len(quantiles)
        self.horizons = [int(h) for h in horizons]
        self.H = len(self.horizons)
        self.train_gat_heads = train_gat_heads
        self.transformer_batch_first = transformer_batch_first

        self.seq = seq
        self.pred_len = pred_len
        self.alpha = 0.5
        self.a_sparse = a_sparse
        self.nodes = a_sparse.shape[0]

        # ===== Backbone 不动 =====
        self.gat_layer = MultiHeadsGATLayer(
            a_sparse,
            seq,
            seq,
            head_n=4,
            dropout=0,
            alpha=0.2,
            train_heads=train_gat_heads,
        )
        self.gcn = nn.Linear(in_features=seq, out_features=seq)

        self.informer = Informer(
            input_dim=seq * 2,
            seq_len=seq,
            pred_len=pred_len,
            d_model=hidden_dim,
            batch_first=transformer_batch_first,
        )

        self.dropout = nn.Dropout(p=0.5)
        self.leakyrelu = nn.LeakyReLU()

        # ⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐
        # Quantile Head
        # ⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐

        self.quantile_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.H * self.Q)  # base + increments per horizon
        )

        self.softplus = nn.Softplus()

    def forward(self, occ, prc):
        b, n, s = occ.shape

        data = torch.stack([occ, prc], dim=3).reshape(2 * b, n, s)

        atts_mat = self.gat_layer(data)
        occ_conv1 = torch.matmul(atts_mat, data)
        occ_conv1 = self.dropout(self.leakyrelu(self.gcn(occ_conv1)))

        atts_mat2 = self.gat_layer(occ_conv1)
        occ_conv2 = torch.matmul(atts_mat2, occ_conv1)
        occ_conv2 = self.dropout(self.leakyrelu(self.gcn(occ_conv2)))

        occ_conv1 = (1 - self.alpha) * occ_conv1 + self.alpha * data
        occ_conv2 = (1 - self.alpha) * occ_conv2 + self.alpha * occ_conv1

        occ_conv2 = occ_conv2.view(b, n, s, -1)

        # informer 输出: (B, N, pred_len, hidden_dim)
        y = self.informer(occ_conv2)

        raw_q = self.quantile_head(y)
        horizon_count = int(getattr(self, "H", 1))
        quantile_count = int(getattr(self, "Q", raw_q.shape[-1]))
        if horizon_count > 1:
            raw_q = raw_q.view(b, n, horizon_count, quantile_count)  # (B,N,H,Q)

        # ===== 单调构造 =====

        base = self.softplus(raw_q[..., 0:1])  # >=0

        increments = self.softplus(raw_q[..., 1:])  # >=0

        quantiles = torch.cat(
            [base, base + torch.cumsum(increments, dim=-1)],
            dim=-1
        )

        return quantiles


class TemporalGraphQuantile(nn.Module):
    """
    Batch-safe temporal-spatial quantile forecaster.

    The temporal encoder runs along each node's own history, then graph diffusion
    shares information across nodes. This avoids the legacy Informer behavior
    where different sliding-window samples could attend to each other through the
    batch axis.
    """
    def __init__(
        self,
        a_sparse,
        seq=24,
        hidden_dim=128,
        quantiles=None,
        horizons=None,
        input_features=2,
        temporal_layers=1,
        graph_layers=2,
        dropout=0.1,
        graph_self_loop=True,
    ):
        super().__init__()

        if quantiles is None:
            quantiles = [0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95]
        if horizons is None:
            horizons = [1]

        self.quantiles = [float(q) for q in quantiles]
        self.Q = len(self.quantiles)
        self.horizons = [int(h) for h in horizons]
        self.H = len(self.horizons)
        self.seq = seq
        self.hidden_dim = hidden_dim
        self.graph_layers = graph_layers
        self.graph_self_loop = graph_self_loop

        adj_dense = a_sparse.to_dense().float()
        adj_dense = torch.nan_to_num(adj_dense, nan=0.0, posinf=0.0, neginf=0.0)
        adj_dense = torch.clamp(adj_dense, min=0.0)
        if graph_self_loop:
            adj_dense = adj_dense.clone()
            adj_dense.fill_diagonal_(1.0)
        row_sum = adj_dense.sum(dim=1, keepdim=True).clamp_min(1e-6)
        self.register_buffer("adj_norm", (adj_dense / row_sum).to_sparse().coalesce())

        self.feature_proj = nn.Sequential(
            nn.Linear(input_features, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.temporal_encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=temporal_layers,
            batch_first=True,
            dropout=dropout if temporal_layers > 1 else 0.0,
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)

        self.graph_linears = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(graph_layers)
        ])
        self.graph_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(graph_layers)
        ])

        self.horizon_embedding = nn.Embedding(self.H, hidden_dim)
        self.quantile_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.Q),
        )
        self.dropout = nn.Dropout(dropout)
        self.softplus = nn.Softplus()
        self._reset_quantile_head()

    @staticmethod
    def _softplus_inverse(value: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.expm1(value).clamp_min(1e-8))

    def _reset_quantile_head(self) -> None:
        last_layer = self.quantile_head[-1]
        nn.init.zeros_(last_layer.weight)

        quantiles = torch.tensor(self.quantiles, dtype=torch.float32)
        initial_curve = 0.02 + 0.18 * quantiles
        increments = torch.empty_like(initial_curve)
        increments[0] = initial_curve[0]
        increments[1:] = initial_curve[1:] - initial_curve[:-1]
        with torch.no_grad():
            last_layer.bias.copy_(self._softplus_inverse(increments))

    def forward(self, occ, prc):
        b, n, s = occ.shape
        x = torch.stack([occ, prc], dim=-1)  # (B,N,S,2)
        x = self.feature_proj(x)
        x = x.reshape(b * n, s, -1)

        _, h = self.temporal_encoder(x)
        h = h[-1].view(b, n, self.hidden_dim)
        h = self.temporal_norm(h)

        for linear, norm in zip(self.graph_linears, self.graph_norms):
            msg = h.transpose(0, 1).reshape(n, b * self.hidden_dim)
            msg = torch.sparse.mm(self.adj_norm.to(dtype=h.dtype, device=h.device), msg)
            msg = msg.reshape(n, b, self.hidden_dim).transpose(0, 1)
            msg = self.dropout(F.gelu(linear(msg)))
            h = norm(h + msg)

        horizon_ids = torch.arange(self.H, device=h.device)
        horizon_h = h.unsqueeze(2) + self.horizon_embedding(horizon_ids).view(1, 1, self.H, -1)
        raw_q = self.quantile_head(horizon_h)

        base = self.softplus(raw_q[..., 0:1])
        increments = self.softplus(raw_q[..., 1:])
        quantiles = torch.cat(
            [base, base + torch.cumsum(increments, dim=-1)],
            dim=-1,
        )
        return quantiles


class MultiScaleTemporalGraphQuantile(nn.Module):
    """
    Multi-scale extension of TemporalGraphQuantile.

    A short temporal branch focuses on recent dynamics, while a long branch sees
    the full input window. A horizon-conditioned gate learns how much each
    forecast horizon should rely on the short or long representation.
    """
    def __init__(
        self,
        a_sparse,
        seq=48,
        short_seq=24,
        hidden_dim=128,
        quantiles=None,
        horizons=None,
        input_features=2,
        temporal_layers=1,
        graph_layers=2,
        dropout=0.1,
        graph_self_loop=True,
    ):
        super().__init__()

        if quantiles is None:
            quantiles = [0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95]
        if horizons is None:
            horizons = [1]

        self.quantiles = [float(q) for q in quantiles]
        self.Q = len(self.quantiles)
        self.horizons = [int(h) for h in horizons]
        self.H = len(self.horizons)
        self.seq = int(seq)
        self.short_seq = min(int(short_seq), self.seq)
        self.hidden_dim = hidden_dim
        self.graph_layers = graph_layers
        self.graph_self_loop = graph_self_loop

        adj_dense = a_sparse.to_dense().float()
        adj_dense = torch.nan_to_num(adj_dense, nan=0.0, posinf=0.0, neginf=0.0)
        adj_dense = torch.clamp(adj_dense, min=0.0)
        if graph_self_loop:
            adj_dense = adj_dense.clone()
            adj_dense.fill_diagonal_(1.0)
        row_sum = adj_dense.sum(dim=1, keepdim=True).clamp_min(1e-6)
        self.register_buffer("adj_norm", (adj_dense / row_sum).to_sparse().coalesce())

        self.feature_proj = nn.Sequential(
            nn.Linear(input_features, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.short_encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=temporal_layers,
            batch_first=True,
            dropout=dropout if temporal_layers > 1 else 0.0,
        )
        self.long_encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=temporal_layers,
            batch_first=True,
            dropout=dropout if temporal_layers > 1 else 0.0,
        )
        self.short_norm = nn.LayerNorm(hidden_dim)
        self.long_norm = nn.LayerNorm(hidden_dim)

        self.graph_linears = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(graph_layers)
        ])
        self.graph_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(graph_layers)
        ])

        self.horizon_embedding = nn.Embedding(self.H, hidden_dim)
        self.horizon_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.quantile_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.Q),
        )
        self.dropout = nn.Dropout(dropout)
        self.softplus = nn.Softplus()
        self._reset_quantile_head()

    @staticmethod
    def _softplus_inverse(value: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.expm1(value).clamp_min(1e-8))

    def _reset_quantile_head(self) -> None:
        last_layer = self.quantile_head[-1]
        nn.init.zeros_(last_layer.weight)

        quantiles = torch.tensor(self.quantiles, dtype=torch.float32)
        initial_curve = 0.02 + 0.18 * quantiles
        increments = torch.empty_like(initial_curve)
        increments[0] = initial_curve[0]
        increments[1:] = initial_curve[1:] - initial_curve[:-1]
        with torch.no_grad():
            last_layer.bias.copy_(self._softplus_inverse(increments))

    def _graph_diffuse(self, h: torch.Tensor) -> torch.Tensor:
        b, n, _ = h.shape
        for linear, norm in zip(self.graph_linears, self.graph_norms):
            msg = h.transpose(0, 1).reshape(n, b * self.hidden_dim)
            msg = torch.sparse.mm(self.adj_norm.to(dtype=h.dtype, device=h.device), msg)
            msg = msg.reshape(n, b, self.hidden_dim).transpose(0, 1)
            msg = self.dropout(F.gelu(linear(msg)))
            h = norm(h + msg)
        return h

    def forward(self, occ, prc):
        b, n, s = occ.shape
        x = torch.stack([occ, prc], dim=-1)  # (B,N,S,2)
        x = self.feature_proj(x)

        long_x = x.reshape(b * n, s, -1)
        _, long_h = self.long_encoder(long_x)
        long_h = self.long_norm(long_h[-1].view(b, n, self.hidden_dim))

        short_steps = min(self.short_seq, s)
        short_x = x[:, :, -short_steps:, :].reshape(b * n, short_steps, -1)
        _, short_h = self.short_encoder(short_x)
        short_h = self.short_norm(short_h[-1].view(b, n, self.hidden_dim))

        short_h = self._graph_diffuse(short_h)
        long_h = self._graph_diffuse(long_h)

        horizon_ids = torch.arange(self.H, device=x.device)
        horizon_e = self.horizon_embedding(horizon_ids).view(1, 1, self.H, -1)
        short_h = short_h.unsqueeze(2).expand(-1, -1, self.H, -1)
        long_h = long_h.unsqueeze(2).expand(-1, -1, self.H, -1)
        horizon_e = horizon_e.expand(b, n, -1, -1)

        gate = self.horizon_gate(torch.cat([short_h, long_h, horizon_e], dim=-1))
        horizon_h = gate * short_h + (1.0 - gate) * long_h + horizon_e
        raw_q = self.quantile_head(horizon_h)

        base = self.softplus(raw_q[..., 0:1])
        increments = self.softplus(raw_q[..., 1:])
        quantiles = torch.cat(
            [base, base + torch.cumsum(increments, dim=-1)],
            dim=-1,
        )
        return quantiles


class PAGInformerQuantileIndependent(nn.Module):
    """
    Ablation A: independent quantile outputs without the monotonic cumulative head.
    The backbone and quantile head are kept identical to PAGInformerQuantile.
    Each quantile is passed through an element-wise Softplus to preserve non-negativity,
    but no cross-quantile ordering constraint is imposed.
    """
    def __init__(self, a_sparse, seq=12, pred_len=6, hidden_dim=128, quantiles=None):
        super(PAGInformerQuantileIndependent, self).__init__()

        if quantiles is None:
            quantiles = [0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95]

        self.quantiles = quantiles
        self.Q = len(quantiles)

        self.seq = seq
        self.pred_len = pred_len
        self.alpha = 0.5
        self.a_sparse = a_sparse
        self.nodes = a_sparse.shape[0]

        self.gat_layer = MultiHeadsGATLayer(a_sparse, seq, seq, head_n=4, dropout=0, alpha=0.2)
        self.gcn = nn.Linear(in_features=seq, out_features=seq)

        self.informer = Informer(
            input_dim=seq * 2,
            seq_len=seq,
            pred_len=pred_len,
            d_model=hidden_dim
        )

        self.dropout = nn.Dropout(p=0.5)
        self.leakyrelu = nn.LeakyReLU()

        self.quantile_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.Q)
        )

        self.softplus = nn.Softplus()

    def forward(self, occ, prc):
        b, n, s = occ.shape

        data = torch.stack([occ, prc], dim=3).reshape(2 * b, n, s)

        atts_mat = self.gat_layer(data)
        occ_conv1 = torch.matmul(atts_mat, data)
        occ_conv1 = self.dropout(self.leakyrelu(self.gcn(occ_conv1)))

        atts_mat2 = self.gat_layer(occ_conv1)
        occ_conv2 = torch.matmul(atts_mat2, occ_conv1)
        occ_conv2 = self.dropout(self.leakyrelu(self.gcn(occ_conv2)))

        occ_conv1 = (1 - self.alpha) * occ_conv1 + self.alpha * data
        occ_conv2 = (1 - self.alpha) * occ_conv2 + self.alpha * occ_conv1

        occ_conv2 = occ_conv2.view(b, n, s, -1)
        y = self.informer(occ_conv2)

        raw_q = self.quantile_head(y)
        return self.softplus(raw_q)

class PAGInformerQ50MSE(nn.Module):
    """
    q0.5-only point forecasting model trained with MSE.
    Backbone same as PAGInformerQuantile, but the head outputs a single value.
    """
    def __init__(self, a_sparse, seq=12, pred_len=6, hidden_dim=128, nonneg_output=True):
        super(PAGInformerQ50MSE, self).__init__()

        self.seq = seq
        self.pred_len = pred_len
        self.alpha = 0.5
        self.a_sparse = a_sparse
        self.nodes = a_sparse.shape[0]
        self.nonneg_output = nonneg_output

        # ===== Backbone 不动 =====
        self.gat_layer = MultiHeadsGATLayer(a_sparse, seq, seq, head_n=4, dropout=0, alpha=0.2)
        self.gcn = nn.Linear(in_features=seq, out_features=seq)

        self.informer = Informer(
            input_dim=seq * 2,
            seq_len=seq,
            pred_len=pred_len,
            d_model=hidden_dim
        )

        self.dropout = nn.Dropout(p=0.5)
        self.leakyrelu = nn.LeakyReLU()

        # ===== q0.5 Head：只输出 1 个值 =====
        self.q50_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # only one output: "q0.5"
        )

        self.softplus = nn.Softplus()

    def forward(self, occ, prc):
        b, n, s = occ.shape

        data = torch.stack([occ, prc], dim=3).reshape(2 * b, n, s)

        atts_mat = self.gat_layer(data)
        occ_conv1 = torch.matmul(atts_mat, data)
        occ_conv1 = self.dropout(self.leakyrelu(self.gcn(occ_conv1)))

        atts_mat2 = self.gat_layer(occ_conv1)
        occ_conv2 = torch.matmul(atts_mat2, occ_conv1)
        occ_conv2 = self.dropout(self.leakyrelu(self.gcn(occ_conv2)))

        occ_conv1 = (1 - self.alpha) * occ_conv1 + self.alpha * data
        occ_conv2 = (1 - self.alpha) * occ_conv2 + self.alpha * occ_conv1

        occ_conv2 = occ_conv2.view(b, n, s, -1)

        # informer 输出: (B, N, pred_len, hidden_dim)
        y = self.informer(occ_conv2)

        # head 输出: (B, N, pred_len, 1)  (或 (B,N,1) 取决于 informer实现)
        pred = self.q50_head(y)

        # 统一成 (B,N)（你当前 pred_len=1，训练脚本也期待 (B,N)）
        # 如果你未来 pred_len>1，可改成 return pred.squeeze(-1) 保留 (B,N,pred_len)
        pred = pred[..., -1, 0] if pred.dim() == 4 else pred[..., 0]  # 兼容两种形状

        if self.nonneg_output:
            pred = self.softplus(pred)

        return pred
