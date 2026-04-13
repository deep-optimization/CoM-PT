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
    --epochs 22 \
    --workers=8 \
    --model ViT-M-16 \
    --t-model ViT-S-16 \
    --t-model-checkpoint /mnt/workspace/code/CoM/src/logs/2025_04_23-16_26_38-model_ViT-S-16-lr_0.001-b_128-epochs_64-tag_cc3m-cc12m-baseline-e64/checkpoints/baseliine_vit-s-16_cc3m_cc12m.pth \
    --logs logs/ \
    --alpha_fd_loss 500. \
    --weight-init True \
    --use-longcap True \
    --tag merged-15m-com 
