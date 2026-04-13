cd src
NCCL_P2P_DISABLE=1 torchrun --nproc_per_node 8 -m \
    --master_addr=127.0.0.2 --master_port=29533 \
    training.main_kd \
    --save-frequency 2 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data="dataset/csv/cc3m_lc.csv,dataset/csv/cc12m_lc.csv"  \
    --val-data="dataset/csv/cc3m_val.csv"  \
    --data-root dataset/cc3m/,dataset/ \
    --val-data-root dataset/cc3m/ \
    --csv-img-key filepath \
    --csv-caption-key title \
    --imagenet-val=dataset/imagenet/val/ \
    --warmup 1000 \
    --batch-size=128 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs 20 \
    --workers=8 \
    --model ViT-B-16 \
    --t-model ViT-M-16 \
    --t-model-checkpoint /mnt/workspace/code/CoM/src/logs/2025_07_21-06_02_51-t_model_ViT-S-16-s_model_ViT-M-16-lr_0.001-b_128-tag_com/checkpoints/com_vit-m-16_e22.pt \
    --alpha_fd_loss 500. \
    --weight-init True \
    --use-longcap True \
    --logs new_logs/ \
    --tag merged-15m-com 
