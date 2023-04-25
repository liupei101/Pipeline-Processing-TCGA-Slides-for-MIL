#!/bin/bash
set -e

# Sample patches of SIZE x SIZE at LEVEL 
LEVEL=1
SIZE=256

# Path where CLAM is installed
DIR_REPO=../CLAM

# Root path to pathology images 
DIR_READ=/NAS02/RawData/tcga_rcc
DIR_SAVE=/NAS02/ExpData/tcga_rcc

cd ${DIR_REPO}

echo "run seg & patching for all slides"
CUDA_VISIBLE_DEVICES=1 python3 create_patches_fp.py \
    --source ${DIR_READ} \
    --save_dir ${DIR_SAVE}/tiles-l${LEVEL}-s${SIZE} \
    --patch_size ${SIZE} \
    --step_size ${SIZE} \
    --preset tcga.csv \
    --patch_level ${LEVEL} \
    --seg --patch --stitch \
    --no_auto_skip --in_child_dir