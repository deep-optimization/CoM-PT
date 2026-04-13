cd src
torchrun --nproc_per_node 8 -m \
    --master_addr=127.0.0.2 --master_port=29533 \
    training.main_kd \
    --save-frequency 16 \
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
    --model ViT-B-16 \
    --t-model ViT-S-16 \
    --t-model-checkpoint ./pretrained_models/com/com_vit-s_e24-linear.pt \
    --logs logs/ \
    --alpha_fd_loss 500. \
    --weight-init True \
    --use-longcap True \
    --tag com 
