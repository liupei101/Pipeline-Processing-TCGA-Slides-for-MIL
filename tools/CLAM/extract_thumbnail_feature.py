import torch
import torch.nn as nn
from math import floor
import os
import random
import numpy as np
import pdb
import time
from datasets.dataset_h5 import Dataset_All_Bags, colnor_followed_transforms
from torch.utils.data import DataLoader
from models.resnet_custom import resnet50_baseline, resnet50_full, resnet50_MoCo, resnet18_ST
import argparse
from utils.utils import print_network, collate_features
from utils.utils import get_slide_id, get_slide_fullpath, get_color_normalizer
from utils.utils import color_normalization
from utils.file_utils import save_hdf5
import h5py
import openslide
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def compute_thumbnail_feature(output_path, wsi_object, model, 
    target_patch_size=-1, color_normalizer=None, roi_transforms=None):
    
    if roi_transforms is None:
        roi_transforms = colnor_followed_transforms(pretrained=True)

    img = wsi_object.get_thumbnail((256, 256))

    if color_normalizer is not None:
        img = np.array(img).astype(np.uint8)
        img = color_normalization(img, color_normalizer)
        
    data = roi_transforms(img).unsqueeze(0) # torch.Tensor

    mode = 'w'
    with torch.no_grad():
        data = data.to(device, non_blocking=True)
        
        features = model(data)
        
        features = features.cpu().numpy()

        asset_dict = {'features': features, 'image_type': np.array([-1]), 'patch_size': np.array([256, 256])}
        save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
        mode = 'a'

parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--arch', type=str, default='RN50B', help='Choose which architecture to use for extracting features.')
parser.add_argument('--ckpt_path', type=str, default=None, help='The checkpoint path for pretrained models.')
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--slide_in_child_dir', default=False, action='store_true')
parser.add_argument('--color_norm', default=None, type=str, help='use x5-256 or x20-256 or thumbnail-256')
args = parser.parse_args()


if __name__ == '__main__':

    print('initializing dataset')
    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError('No csv_path is gotten.')

    bags_dataset = Dataset_All_Bags(csv_path)
    
    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
    dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

    color_normalizer = None
    if args.color_norm is not None:
        # color normalization for pathology images
        color_normalizer = get_color_normalizer(args.color_norm)

    print('loading model checkpoint of arch {} from {}'.format(args.arch, args.ckpt_path))
    if args.arch == 'RN50-B':
        model = resnet50_baseline(pretrained=True)
    elif args.arch == 'RN50-F':
        model = resnet50_full(pretrained=True)
    elif args.arch == 'RN50-MoCo-B':
        model = resnet50_MoCo('baseline', ckpt_from=args.ckpt_path) # 1024-d from layer3 of ResNet50
    elif args.arch == 'RN50-MoCo-F':
        model = resnet50_MoCo('full', ckpt_from=args.ckpt_path) # 2048-d from layer4 of ResNet50
    elif args.arch == 'RN18-SimCL-TCGA':
        model = resnet18_ST(ckpt_from=args.ckpt_path) # 512-d from layer4 of ResNet18
    else:
        raise NotImplementedError("Please specify a valid architecture.")
    model = model.to(device)
    
    # print_network(model)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    model.eval()
    total = len(bags_dataset)

    for bag_candidate_idx in range(total):
        slide_name = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
        slide_id = get_slide_id(slide_name, has_ext=False, in_child_dir=args.slide_in_child_dir)
        bag_name = slide_id+'.h5'
        # h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
        
        # if not os.path.exists(h5_file_path):
        #     print('skiped slide {}, h5 file not found'.format(slide_id))
        #     continue
        
        slide_file_path = get_slide_fullpath(args.data_slide_dir, slide_name, 
            in_child_dir=args.slide_in_child_dir) + args.slide_ext
        print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
        print(slide_id)

        if not args.no_auto_skip and slide_id+'.pt' in dest_files:
            print('skipped {}'.format(slide_id))
            continue 

        output_file_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
        time_start = time.time()
        wsi = openslide.open_slide(slide_file_path)
        compute_thumbnail_feature(output_file_path, wsi, model, 
            target_patch_size=args.target_patch_size,
            color_normalizer=color_normalizer
        )
        wsi.close()
        # output_file_path = compute_w_loader(h5_file_path, output_file_path, wsi, 
        # model = model, batch_size = args.batch_size, verbose = 1, print_every = 20, 
        # custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size,
        # sampler_setting=args_sampler, color_normalizer=color_normalizer)
        time_elapsed = time.time() - time_start
        print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
        file = h5py.File(output_file_path, "r")

        features = file['features'][:]
        print('features size: ', features.shape)
        # print('coordinates size: ', file['coords'].shape)
        features = torch.from_numpy(features)
        bag_base, _ = os.path.splitext(bag_name)
        torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))
        file.close()

