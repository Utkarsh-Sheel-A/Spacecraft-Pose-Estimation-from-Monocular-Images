
#!/bin/bash
conda activate speedplus
python /home/dex/speedplusbaseline/train.py \
    --savedir '/home/dex/speedplusbaseline/checkpoints/krn/synthetic_only' \
    --logdir '/home/dex/speedplusbaseline/log/krn/synthetic_only' \
    --model_name 'krn' \
    --input_shape 224 224 \
    --batch_size 48 \
    --max_epochs 75 \
    --optimizer 'adamw' \
    --lr 0.001 \
    --weight_decay 0.01 \
    --lr_decay_alpha 0.95 \
    --train_domain 'synthetic' \
    --test_domain 'synthetic' \
    --train_csv 'train.csv' \
    --test_csv 'test.csv'
