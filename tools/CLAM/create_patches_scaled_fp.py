import argparse
import numpy as np
import pandas as pd
import os
import os.path as osp
from tqdm import tqdm
import h5py
import openslide

from wsi_core.wsi_utils import save_hdf5, StitchCoords

SEP = '--'
"""
path_patchi: level = L,   size = 256
path_patcho: level = L-1, size = 256
path_patchi * patch_scale -> path_patcho
"""
def get_scaled_matrix(width, height, scale=4, downscale=4):
    """
    scale x scale patches with a size of width x height.
    downscale is the current patch's downscale.
    """
    # back to ref size
    width = width * downscale
    height = height * downscale
    mat = np.zeros((scale, scale, 2), dtype=np.int32)
    for j in range(scale):
        for i in range(scale):
            mat[j][i] = np.array([i * width, j * height])
    mat = np.reshape(mat, (-1, 2))
    return mat

def get_scaled_attrs(origin_attrs, scale=4):
    attrs = {
        'downsample': origin_attrs['downsample'] / scale,
        'downsampled_level_dim': origin_attrs['downsampled_level_dim'] * scale,
        'level_dim': origin_attrs['level_dim'] * scale,
        'name': origin_attrs['name'],
        'patch_level': origin_attrs['patch_level'] - 1,
        'patch_size': origin_attrs['patch_size'],
    }
    return attrs

def process_patches(path_patchi, path_patcho, patch_scale):
    scaled_coords = np.zeros((1,2), dtype=np.int32)
    scaled_attrs  = None

    with h5py.File(path_patchi, 'r') as hf:
        data_coords = hf['coords']
        scaled_attrs = get_scaled_attrs(data_coords.attrs, patch_scale)

        psize = data_coords.attrs['patch_size']
        pdownscale = round(data_coords.attrs['downsample'][0] / patch_scale) # output patch downscale
        scaled_mat = get_scaled_matrix(psize, psize, patch_scale, downscale=pdownscale)
        coords = data_coords[:]
        for coord in coords:
            cur_coords = scaled_mat + coord
            scaled_coords = np.concatenate((scaled_coords, cur_coords), axis=0)

    scaled_coords = scaled_coords[1:] # ignore the first row
    scaled_attrs['save_path'] = osp.dirname(path_patcho)
    save_hdf5(path_patcho, {'coords': scaled_coords}, {'coords': scaled_attrs}, mode='w')

def create_patches(input_dir, path_process_list, save_dir, patch_scale, auto_skip=False):
    process_list = pd.read_csv(path_process_list)
    for i in tqdm(process_list.index):
        slide_id = process_list.loc[i, 'slide_id']
        status = process_list.loc[i, 'process']
        if status != 0:
            print(f'skipped the file {slide_id} not processed')
            continue

        slide_id = slide_id.split(SEP)[1][:-4]
        path_origin_patch = osp.join(input_dir, slide_id + '.h5')
        path_save_patch = osp.join(save_dir, slide_id + '.h5')

        if not osp.exists(path_origin_patch):
            print(f'skipped the file {path_origin_patch} not found')
            continue

        if auto_skip and osp.exists(path_save_patch):
            print(f'skipped the file {path_save_patch} processed')
            continue

        process_patches(path_origin_patch, path_save_patch, patch_scale)

def process_stitch(path_wsi, path_patch, path_save):
    """
    path_wsi: path to WSI
    path_patch: path to the patch sliced from WSI, saved as a h5 file
    path_save: path to save the stitched picture, saved as a jpg file
    """
    wsi_object = openslide.open_slide(path_wsi)
    heatmap = StitchCoords(path_patch, wsi_object, downscale=64)
    heatmap.save(path_save)

def stitch_patches(wsi_dir, patch_dir, save_dir, path_process_list, auto_skip=False):
    process_list = pd.read_csv(path_process_list)
    for i in tqdm(process_list.index):
        slide_id = process_list.loc[i, 'slide_id']
        status = process_list.loc[i, 'process']

        patient_id = slide_id.split(SEP)[0]
        slide_id = slide_id.split(SEP)[1][:-4]
        path_wsi = osp.join(wsi_dir, patient_id, slide_id + '.svs')
        path_patch = osp.join(patch_dir, slide_id + '.h5')
        path_save = osp.join(save_dir, slide_id + '.jpg')

        if not osp.exists(path_wsi):
            print(f'skipped the WSI {path_wsi} not found')
            continue

        if not osp.exists(path_patch):
            print(f'skipped the patch {path_patch} not found')
            continue

        if auto_skip and osp.exists(path_save):
            print(f'skipped the file {path_save} processed')
            continue

        process_stitch(path_wsi, path_patch, path_save)


parser = argparse.ArgumentParser(description='scaled patch and stitch')
parser.add_argument('--source', type=str, help='path to folder containing raw wsi image files')
parser.add_argument('--input_dir', type=str, help='directory to input data (origin processed patches)')
parser.add_argument('--process_list', type=str, default='process_list_autogen.csv', 
                    help='CSV file that logs the processed WSI')
parser.add_argument('--save_dir', type = str,
                    help='directory to save processed data (scaled patches)')
parser.add_argument('--stitch', default=False, action='store_true')
parser.add_argument('--auto_skip', default=False, action='store_true')
parser.add_argument('--patch_scale', type=int, default=4, 
                    help='scale relative to the patch at input_dir')

if __name__ == '__main__':
    args = parser.parse_args()

    patch_input_dir = osp.join(args.input_dir, 'patches')
    process_list_path = osp.join(args.input_dir, args.process_list)
    patch_save_dir = osp.join(args.save_dir, 'patches')
    stitch_save_dir = osp.join(args.save_dir, 'stitches')
    
    if not osp.exists(patch_save_dir):
        os.makedirs(patch_save_dir)
    if not osp.exists(stitch_save_dir):
        os.makedirs(stitch_save_dir)

    print('source: ', args.source)
    print('patch_input_dir: ', patch_input_dir)
    print('patch_save_dir: ', patch_save_dir)
    print('stitch_save_dir: ', stitch_save_dir)

    create_patches(patch_input_dir, process_list_path, patch_save_dir, args.patch_scale, auto_skip=args.auto_skip)

    if args.stitch:
        stitch_patches(args.source, patch_save_dir, stitch_save_dir, process_list_path)
