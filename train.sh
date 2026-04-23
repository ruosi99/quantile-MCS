python train.py \
    --data_dir "data/datasets/ST_EVCDP_v2/" \
    --model_name dura_pag_informer_quantile_on_pretrain \
    --load_method "models.PAGInformerQuantile(a_sparse=adj_sparse, seq=seq_len).to(device)" \
    --is_train False \
    --is_pre_train True \
    --use_cuda True \
    --batch_size 8

# python train.py \
#     --data_dir "data/datasets/ST_EVCDP_v2/" \
#     --model_name dura_pag_informer_quantile_0.5_MSE_on_pretrain \
#     --load_method "models.PAGInformerQuantile(a_sparse=adj_sparse, seq=seq_len, quantiles=[0.5]).to(device)" \
#     --is_train True \
#     --is_pre_train True \
#     --use_cuda True \
#     --batch_size 8 \
#     --quantiles 0.5

# python train.py \
#     --data_dir "data/datasets/ST_EVCDP_v2/" \
#     --model_name dura_pag_informer_on_pretrain \
#     --load_method "models.PAGInformer(a_sparse=adj_sparse, seq=seq_len).to(device)" \
#     --is_train True \
#     --is_pre_train True \
#     --use_cuda True \
#     --batch_size 8

# python train.py \
#     --data_dir "data/datasets/ST_EVCDP_v2/" \
#     --model_name dura_pag_informer_q50_MSE_on_pretrain \
#     --load_method "models.PAGInformerQ50MSE(a_sparse=adj_sparse, seq=seq_len).to(device)" \
#     --is_train True \
#     --is_pre_train True \
#     --use_cuda True \
#     --batch_size 8

# python train.py \
#     --data_dir "data/datasets/ST_EVCDP_v2/" \
#     --model_name volume_pag_informer_quantile_on_pretrain \
#     --load_method "models.PAGInformerQuantile(a_sparse=adj_sparse, seq=seq_len).to(device)" \
#     --is_train False \
#     --is_pre_train True \
#     --use_cuda True \
#     --batch_size 8