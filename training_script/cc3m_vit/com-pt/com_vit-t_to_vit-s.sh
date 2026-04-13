cd src
NCCL_P2P_DISABLE=1 torchrun --nproc_per_node 8 -m \
    --master_addr=127.0.0.2 --master_port=29533 \
    training.main_kd \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data="dataset/csv/cc3m_lc.csv"  \
    --val-data="dataset/csv/cc3m_val.csv"  \
    --data-root dataset/cc3m/ \
    --val-data-root dataset/cc3m/ \
    --csv-img-key filepath \
    --csv-caption-key title \
    --imagenet-val=dataset/imagenet/val/ \
    --warmup 1000 \
    --batch-size=128 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs 24 \
    --workers=8 \
    --model ViT-S-16 \
    --t-model ViT-T-16-text-w256 \
    --t-model-checkpoint ./pretrained_models/baseline/baseline_vit-t_e128.pt \
    --logs logs/ \
    --alpha_fd_loss 500. \
    --weight-init True \
    --use-longcap True \
    --tag com 
# main_kd_meta
# src/pretrained_models/vit_n_16.pt
# vit_t_16_text_256
# src/pretrained_models/baseline/baseline_vit-t_e128.pt
# /mnt/workspace/code/CoM/src/pretrained_models/baseline/baseline_vit-t_e32.pt
# /mnt/workspace/code/CoM/src/logs/2025_04_17-02_52_26-model_ViT-T-16-text-w256-lr_0.001-b_128-epochs_64-tag_cc3m-cc12m-baseline-e64/checkpoints/baseline_vit-t_e64_cc15m.pth