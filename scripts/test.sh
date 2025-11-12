#!/bin/bash
#
device=0

base_dir='mvtec_train'
cpkt_path=/mnt/afs/yangziteng/project/project/vipclip/models/mvtec_train_epoch_10.pth

script=test.py
output=mvtec_train_test.log

n_ctx=7
n_cls=3
topk=50

CUDA_VISIBLE_DEVICES=${device} python ${script} --dataset visa \
    --data_path /ssd2/yangziteng/anomaly_detect/data/visa --save_path ./final/${base_dir}/zero_shot_visa\
    --checkpoint_path ${cpkt_path} --visualize\
    --features_list 6 12 18 24  --image_size 518 --n_ctx ${n_ctx} --n_cls ${n_cls} --seed 111 --topk ${topk} >> ${output} 2>&1 &

CUDA_VISIBLE_DEVICES=${device} python ${script} --dataset colon \
    --data_path /ssd2/yangziteng/anomaly_detect/data/CVC-ColonDB --save_path ./final/${base_dir}/zero_shot_ndb\
    --checkpoint_path ${cpkt_path} \
    --features_list 6 12 18 24  --image_size 518 --n_ctx ${n_ctx} --n_cls ${n_cls} --seed 111 --topk ${topk} >> ${output} 2>&1 &

CUDA_VISIBLE_DEVICES=${device} python ${script} --dataset colon \
    --data_path /ssd2/yangziteng/anomaly_detect/data/CVC-ClinicDB --save_path ./final/${base_dir}/zero_shot_cdb\
    --checkpoint_path ${cpkt_path} \
    --features_list 6 12 18 24  --image_size 518 --n_ctx ${n_ctx} --n_cls ${n_cls} --seed 111 --topk ${topk} >> ${output} 2>&1 &

CUDA_VISIBLE_DEVICES=${device} python ${script} --dataset head_ct \
    --data_path /ssd2/yangziteng/anomaly_detect/data/head_ct --save_path ./final/${base_dir}/zero_shot_head_ct10\
    --checkpoint_path ${cpkt_path} \
    --features_list 6 12 18 24  --image_size 518 --n_ctx ${n_ctx} --n_cls ${n_cls} --seed 111 --topk ${topk} >> ${output} 2>&1 &

CUDA_VISIBLE_DEVICES=${device} python ${script} --dataset head_ct \
    --data_path /ssd2/yangziteng/anomaly_detect/data/brain_tumor_dataset --save_path ./final/${base_dir}/zero_shot_btd\
    --checkpoint_path ${cpkt_path} \
    --features_list 6 12 18 24  --image_size 518 --n_ctx ${n_ctx} --n_cls ${n_cls} --seed 111 --topk ${topk} >> ${output} 2>&1 &

CUDA_VISIBLE_DEVICES=${device} python ${script} --dataset DAGM_KaggleUpload \
    --data_path /ssd2/yangziteng/anomaly_detect/data/DAGM_KaggleUpload --save_path ./final/${base_dir}/zero_shot_dagm\
    --checkpoint_path ${cpkt_path} \
    --features_list 6 12 18 24  --image_size 518 --n_ctx ${n_ctx} --n_cls ${n_cls} --seed 111 --topk ${topk} >> ${output} 2>&1 &

CUDA_VISIBLE_DEVICES=${device} python ${script} --dataset mvtec \
    --data_path /ssd2/yangziteng/anomaly_detect/data/mvtec --save_path ./final/${base_dir}/zero_shot_mvtec10\
    --checkpoint_path ${cpkt_path} \
    --features_list 6 12 18 24  --image_size 518 --n_ctx ${n_ctx} --n_cls ${n_cls} --seed 111 --topk ${topk} >> ${output} 2>&1 &

CUDA_VISIBLE_DEVICES=${device} python ${script} --dataset btad \
    --data_path /ssd2/yangziteng/anomaly_detect/data/BTech_Dataset_transformed --save_path ./final/${base_dir}/zero_shot_btad\
    --checkpoint_path ${cpkt_path} \
    --features_list 6 12 18 24  --image_size 518 --n_ctx ${n_ctx} --n_cls ${n_cls} --seed 111 --topk ${topk} >> ${output} 2>&1 &

CUDA_VISIBLE_DEVICES=${device} python ${script} --dataset mpdd \
    --data_path /ssd2/yangziteng/anomaly_detect/data/MPDD --save_path ./final/${base_dir}/zero_shot_mpdd2\
    --checkpoint_path ${cpkt_path} \
    --features_list 6 12 18 24  --image_size 518 --n_ctx ${n_ctx} --n_cls ${n_cls} --seed 111 --topk ${topk} >> ${output} 2>&1 &

CUDA_VISIBLE_DEVICES=${device} python ${script} --dataset SDD \
    --data_path /ssd2/yangziteng/anomaly_detect/data/SDD_anomaly_detection --save_path ./final/${base_dir}/zero_shot_sdd2\
    --checkpoint_path ${cpkt_path} \
    --features_list 6 12 18 24  --image_size 518 --n_ctx ${n_ctx} --n_cls ${n_cls} --seed 111 --topk ${topk} >> ${output} 2>&1 &

CUDA_VISIBLE_DEVICES=${device} python ${script} --dataset DTD \
    --data_path /ssd2/yangziteng/anomaly_detect/data/DTD-Synthetic --save_path ./final/${base_dir}/zero_shot_dtd\
    --checkpoint_path ${cpkt_path} \
    --features_list 6 12 18 24  --image_size 518 --n_ctx ${n_ctx} --n_cls ${n_cls} --seed 111 --topk ${topk} >> ${output} 2>&1 &

CUDA_VISIBLE_DEVICES=${device} python ${script} --dataset head_ct \
    --data_path /ssd2/yangziteng/anomaly_detect/data/Br35H --save_path ./final/${base_dir}/zero_shot_br35\
    --checkpoint_path ${cpkt_path} \
    --features_list 6 12 18 24  --image_size 518 --n_ctx ${n_ctx} --n_cls ${n_cls} --seed 111 --topk ${topk} >> ${output} 2>&1 &

CUDA_VISIBLE_DEVICES=${device} python ${script} --dataset colon \
    --data_path /ssd2/yangziteng/anomaly_detect/data/EndoTect_2020_Segmentation_Test_Dataset --save_path ./final/${base_dir}/zero_shot_end\
    --checkpoint_path ${cpkt_path} \
    --features_list 6 12 18 24  --image_size 518 --n_ctx ${n_ctx} --n_cls ${n_cls} --seed 111 --topk ${topk} >> ${output} 2>&1 &

CUDA_VISIBLE_DEVICES=${device} python ${script} --dataset ISBI \
    --data_path /ssd2/yangziteng/anomaly_detect/data/ISIC --save_path ./final/${base_dir}/zero_shot_isic\
    --checkpoint_path ${cpkt_path} \
    --features_list 6 12 18 24  --image_size 518 --n_ctx ${n_ctx} --n_cls ${n_cls} --seed 111 --topk ${topk} >> ${output} 2>&1 &

CUDA_VISIBLE_DEVICES=${device} python ${script} --dataset colon \
    --data_path /ssd2/yangziteng/anomaly_detect/data/Kvasir-SEG --save_path ./final/${base_dir}/zero_shot_Kvasir\
    --checkpoint_path ${cpkt_path} \
    --features_list 6 12 18 24  --image_size 518 --n_ctx ${n_ctx} --n_cls ${n_cls} --seed 111 --topk ${topk} >> ${output} 2>&1 & 

CUDA_VISIBLE_DEVICES=${device} python ${script} --dataset thyroid \
    --data_path /ssd2/yangziteng/anomaly_detect/data/Thyroid_Dataset/tn3k --save_path ./final/${base_dir}/zero_shot_tn3k\
    --checkpoint_path ${cpkt_path} \
    --features_list 6 12 18 24  --image_size 518 --n_ctx ${n_ctx} --n_cls ${n_cls} --seed 111 --topk ${topk} >> ${output} 2>&1 &
