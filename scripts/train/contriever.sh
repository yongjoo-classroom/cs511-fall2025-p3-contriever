#!/bin/bash
TDIR="/home/xz112/contriever/scripts/preprocess/encoded-data"
TRAINDATASETS="${TDIR}/bert-base-uncased/wikipedia_en_20231101_subset"

rmin=0.05 #min crop ratio
rmax=0.5 #max crop ratio
T=0.05
QSIZE=4096 # Reduced queue size
MOM=0.9995
POOL=average
AUG=delete
PAUG=0.1
LC=0.
mo=bert-base-uncased
mp=none

name=$POOL-rmin$rmin-rmax$rmax-T$T-$QSIZE-$MOM-$mo-$AUG-$PAUG

python3 train.py \
        --model_path $mp \
        --sampling_coefficient $LC \
        --retriever_model_id $mo --pooling $POOL \
        --augmentation $AUG --prob_augmentation $PAUG \
        --train_data $TRAINDATASETS --loading_mode split \
        --ratio_min $rmin --ratio_max $rmax --chunk_length 256 \
        --momentum $MOM --queue_size $QSIZE --temperature $T \
        --warmup_steps 100 --total_steps 100000 --lr 0.00005 \
        --name $name \
        --scheduler linear \
        --optim adamw \
        --per_gpu_batch_size 64 \
        --output_dir /home/xz112/contriever/scripts/train/$name 

