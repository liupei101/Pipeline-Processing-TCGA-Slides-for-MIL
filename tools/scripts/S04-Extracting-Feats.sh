#!/bin/bash
set -e

# Sample patches of SIZE x SIZE at LEVEL 
LEVEL=1
SIZE=256

# Path where CLAM is installed
DIR_REPO=../CLAM

# Root path to pathology images 
DIR_RAW_DATA=/NAS02/RawData/tcga_rcc
DIR_EXP_DATA=/NAS02/ExpData/tcga_rcc

# Sub-directory to the patch coordinates generated from S03
SUBDIR_READ=tiles-l${LEVEL}-s${SIZE}

# Sub-directory to the patch features 
SUBDIR_SAVE=feats-l${LEVEL}-s${SIZE}-RN50-color_norm

cd ${DIR_REPO}

echo "running for extracting features from all tiles"
CUDA_VISIBLE_DEVICES=1 python3 extract_features_fp.py \
    --data_h5_dir ${DIR_EXP_DATA}/${SUBDIR_READ} \
    --data_slide_dir ${DIR_RAW_DATA} \
    --csv_path ${DIR_EXP_DATA}/${SUBDIR_READ}/process_list_autogen.csv \
    --feat_dir ${DIR_EXP_DATA}/${SUBDIR_SAVE} \
    --batch_size 128 \
    --slide_ext .svs \
    --color_norm \
    --slide_in_child_dir --no_auto_skip