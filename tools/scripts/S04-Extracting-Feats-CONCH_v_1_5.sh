#!/bin/bash
set -e

######################################################################
# Please carefully read the notes (1,2,3,4,5,6) for a successful run
######################################################################

# Sample patches of SIZE x SIZE at MAG (as used in S03)
# Note 1: following the setting of TITAN, MAG (magnification) should be set to 20
MAG=20
# Note 2: following the setting of TITAN, SIZE (the patch size at 20x) should be set to 512
SIZE=512
# Note 3: following the setting of CONCH_v1.5, TARGET_PATCH_SIZE should be set to 448
TARGET_PATCH_SIZE=448

# Path where CLAM is installed
DIR_REPO=../CLAM

# Root path to pathology images 
DIR_RAW_DATA=/NAS02/RawData/tcga_rcc
DIR_EXP_DATA=/NAS02/ExpData/tcga_rcc

# Sub-directory to the patch coordinates generated from S03
SUBDIR_READ=tiles-${MAG}x-s${SIZE}

# Arch to be used for patch feature extraction (CONCH is strongly recommended)
ARCH=CONCH_v1.5

# Model path
# Note 4: You need to first apply for its access rights via https://huggingface.co/MahmoodLab/TITAN
# and then download the whole repo to your local path, e.g., /path/to/TITAN
# Note 5: /path/to/TITAN/conch_tokenizer.py: modify the code at line 17: "MahmoodLab/TITAN" to "/path/to/TITAN"
# Note 6: /path/to/TITAN/conch_v1_5.py: modify the code at line 687: let checkpoint_path = "/path/to/TITAN"
#         and comment the lines 682-686
MODEL_CKPT=/path/to/TITAN

# Sub-directory to the patch features 
SUBDIR_SAVE=${SUBDIR_READ}/feats-${ARCH}

cd ${DIR_REPO}

echo "running for extracting features from all tiles"
CUDA_VISIBLE_DEVICES=0 python3 extract_features_fp.py \
    --arch ${ARCH} \
    --ckpt_path ${MODEL_CKPT} \
    --data_h5_dir ${DIR_EXP_DATA}/${SUBDIR_READ} \
    --data_slide_dir ${DIR_RAW_DATA} \
    --csv_path ${DIR_EXP_DATA}/${SUBDIR_READ}/process_list_autogen.csv \
    --feat_dir ${DIR_EXP_DATA}/${SUBDIR_SAVE} \
    --target_patch_size ${TARGET_PATCH_SIZE} \
    --batch_size 128 \
    --slide_ext .svs \
    --slide_in_child_dir