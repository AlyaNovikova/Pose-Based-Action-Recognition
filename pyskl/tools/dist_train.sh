#!/usr/bin/env bash

export MASTER_PORT=$((12000 + $RANDOM % 20000))
set -x

CONFIG=$1
GPUS=$2

MKL_SERVICE_FORCE_INTEL=1 PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$MASTER_PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
# Any arguments from the third one are captured by ${@:3}

#bash tools/dist_train.sh configs/posec3d/slowonly_r50_ntu60_xsub/joint.py 1 --validate --test-last --test-best CONFIG=configs/posec3d/slowonly_r50_ntu60_xsub/joint.py
#MKL_SERVICE_FORCE_INTEL=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=12000 tools/train.py configs/posec3d/slowonly_r50_ntu60_xsub/joint.py --launcher pytorch --validate --test-last --test-best
# Any arguments from the third one are captured by ${@:3}
