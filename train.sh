#!/bin/bash

device=0
base_dir='visa_train'
train_script=train.py
test_script=test.py

save_dir=/mnt/afs/yangziteng/project/project/vipclip/saves/${base_dir}
output=visa_train.log

# Define the parameters as variables
epoch=10
n_ctx=7
n_cls=3
seed=111
topks=(50)

(
  for topk in "${topks[@]}"; do 
    echo "===== TRAIN START: $(date) =====" >> ${output}
    CUDA_VISIBLE_DEVICES=${device} python ${train_script} --dataset visa \
      --train_data_path /mnt/afs/yangziteng/project/Anomaly/data1/visa --save_path ${save_dir}_${topk}/ \
      --features_list 6 12 18 24 --image_size 518 \
      --batch_size 8 \
      --print_freq 1 \
      --epoch 10 \
      --save_freq 5 \
      --n_ctx ${n_ctx} --n_cls ${n_cls} --topk ${topk} --seed ${seed} >> ${output} 2>&1
    wait
  done

  for topk in "${topks[@]}"; do 
    echo "===== TEST START: epoch ${epoch} — $(date) =====" >> ${output}
    CUDA_VISIBLE_DEVICES=${device} python ${test_script} --dataset mvtec \
      --data_path /mnt/afs/yangziteng/project/Anomaly/data1/mvtec \
      --save_path ./final/${base_dir}_${topk}/zero_shot_${epoch} \
      --checkpoint_path ${save_dir}_${topk}/epoch_${epoch}.pth \
      --features_list 6 12 18 24 --image_size 518 \
      --n_ctx ${n_ctx} --n_cls ${n_cls} --topk ${topk} --seed ${seed} >> ${output} 2>&1 &
    wait
  done

  echo "===== FINISHED ALL: $(date) =====" >> ${output}
) &