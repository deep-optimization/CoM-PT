cd src
torchrun --nproc_per_node 8 -m \
    --master_addr=127.0.0.2 --master_port=29533 \
    training.main_kd \
    --save-frequency 12 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data="dataset/csv/cc3m_lc.csv"  \
    --val-data="dataset/csv/cc3m_val.csv"  \
    --data-root dataset/cc3m/ \
    --val-data-root dataset/cc3m/ \
    --csv-img-key filepath \
    --csv-caption-key title \
    --imagenet-val=dataset/imagenet/val/\
    --warmup 1000 \
    --batch-size=128 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs 16 \
    --workers=8 \
    --model ViT-L-16 \
    --t-model ViT-B-16 \
    --t-model-checkpoint ./pretrained_models/com/com_vit-b_e16-linear.pt \
    --logs logs/ \
    --alpha_fd_loss 500. \
    --differ-width True \
    --weight-init True \
    --use-longcap True \
    --tag com 
# main_kd_meta
# src/pretrained_models/vit_n_16.pt
# vit_t_16_text_256
# src/pretrained_models/baseline/baseline_vit-t_e128.pt
# com_vit-b_e16-linear.pt
# /mnt/workspace/code/CoM/src/pretrained_models/swin/com_swin-base_e16.pth