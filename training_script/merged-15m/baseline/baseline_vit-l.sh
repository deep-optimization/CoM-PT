cd src
torchrun --nproc_per_node 8 -m \
    --master_addr=127.0.0.2 --master_port=29513 \
    training.main \
    --save-frequency 4 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data="dataset/csv/cc3m_lc.csv,dataset/csv/cc12m_lc.csv"  \
    --val-data="dataset/csv/cc3m_val.csv"  \
    --data-root dataset/cc3m/,dataset/ \
    --val-data-root dataset/cc3m/ \
    --csv-img-key filepath \
    --csv-caption-key title \
    --imagenet-val=dataset/imagenet/val/ \
    --warmup 8000 \
    --use-longcap True \
    --batch-size=128 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs 64 \
    --workers=8 \
    --model ViT-L-16 \
    --logs logs/ \
    --tag merged-15m-baseline-e64

