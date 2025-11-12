#! /usr/bin/bash
# PARTITION="r1-m1-large" #AMP
# PARTITION="cc4cdf6c-944a-4c46-a9de-3d508a06c4dd"
PARTITION=h100-share2
# PARTITION=b7c081ea-ab5a-4278-ab4a-c51bc222de13
WORKSPACE=a58d023b-de76-475f-89c2-7e50f7aa3c7a
CONTAINTER=registry.ms-sc-01.maoshanwangtech.com/ccr_2/verl:sandbox-20250609205214
# CONTAINTER=registry.ms-sc-01.maoshanwangtech.com/studio-aicl/ubuntu20.04-py3.10-cuda11.8-cudnn8-transformer4.28.0:master-20230626-172512-32302
MOUNT=1f29056c-c3f2-11ee-967e-2aea81fd34ba:/mnt/afs2,047443d2-c3f2-11ee-a5f9-9e29792dec2f:/mnt/afs1,ce3b1174-f6eb-11ee-a372-82d352e10aed:/mnt/afs


#############################
JOBNAME="free_kill_deepeys_debug"
GPU_PER_NODE=1
NNODES=1
SCRIPT=$(dirname "$(readlink -f "$0")")/train.sh
if [[ $PARTITION == "r1-m1"* ]]; then
  SPEC=N6lS.Iq.I10.${GPU_PER_NODE}
else
  SPEC=N6lS.Iu.I80.${GPU_PER_NODE}
fi

#############################


GPUS=$((GPU_PER_NODE * NNODES))

sco acp jobs create \
--workspace-name $WORKSPACE -p $PARTITION \
--container-image-url $CONTAINTER \
--storage-mount $MOUNT \
--worker-spec ${SPEC} \
-f pt -N ${NNODES} \
-j $JOBNAME \
--command="NNODES=$NNODES GPU_PER_NODE=$GPU_PER_NODE SCRIPT=$SCRIPT bash $SCRIPT"
# --command="sleep 1d"
# --priority highest \