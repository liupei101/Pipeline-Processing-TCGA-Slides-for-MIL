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

# Arch to be used for patch feature extraction (CONCH is strongly recommended)
ARCH=PLIP

# Model path
# You need to download the whole project from https://huggingface.co/vinid/plip
MODEL_CKPT=/path/to/vinid/plip

# Sub-directory to the patch features 
SUBDIR_SAVE=feats-l${LEVEL}-s${SIZE}-${ARCH}

cd ${DIR_REPO}

echo "running for extracting features from all tiles"
CUDA_VISIBLE_DEVICES=0 python3 extract_features_fp.py \
    --arch ${ARCH} \
    --ckpt_path ${MODEL_CKPT} \
    --data_h5_dir ${DIR_EXP_DATA}/${SUBDIR_READ} \
    --data_slide_dir ${DIR_RAW_DATA} \
    --csv_path ${DIR_EXP_DATA}/${SUBDIR_READ}/process_list_autogen.csv \
    --feat_dir ${DIR_EXP_DATA}/${SUBDIR_SAVE} \
    --batch_size 128 \
    --slide_ext .svs \
    --slide_in_child_dir \
    --proj_to_contrast N
