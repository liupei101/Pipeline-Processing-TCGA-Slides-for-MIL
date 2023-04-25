import torch
import torch.nn as nn
from math import floor
import os
import random
import numpy as np
import pdb
import time
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
from models.resnet_custom import resnet50_baseline, resnet50_full, resnet50_MoCo, resnet18_ST
from models.vit_hipt import get_vit256
import argparse
from utils.utils import print_network, collate_features
from utils.utils import get_slide_id, get_slide_fullpath
from utils.utils import get_color_normalizer, get_color_augmenter
from utils.file_utils import save_hdf5
from PIL import Image
import h5py
import openslide
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def compute_w_loader(file_path, output_path, wsi, model,
    batch_size = 8, verbose = 0, print_every=20, pretrained=True, 
    custom_downsample=1, target_patch_size=-1, sampler_setting=None, 
    color_normalizer=None, color_augmenter=None, add_patch_noise=None, save_h5_path=None):
    """
    args:
        file_path: directory of bag (.h5 file)
        output_path: directory to save computed features (.h5 file)
        model: pytorch model
        batch_size: batch_size for computing features in batches
        verbose: level of feedback
        pretrained: use weights pretrained on imagenet
        custom_downsample: custom defined downscale factor of image patches
        target_patch_size: custom defined, rescaled image size before embedding
        sampler_setting: custom defined, samlping settings
        color_normalizer: normalization for color space of pathology images
        color_augmenter: color augmentation for patch images
        add_patch_noise: adding noise to patch images
        save_h5_path: path to save features as h5 files
    """
    dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, 
        custom_downsample=custom_downsample, target_patch_size=target_patch_size, 
        sampler_setting=sampler_setting, color_normalizer=color_normalizer, 
        color_augmenter=color_augmenter, add_patch_noise=add_patch_noise)
    kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}   #num_workers 4->0
    loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

    if verbose > 0:
        print('processing {}: total of {} batches'.format(file_path,len(loader)))

    all_feats = None
    all_coors = None
    for count, (batch, coords) in enumerate(loader):
        coords = torch.from_numpy(coords)
        with torch.no_grad():   
            if count % print_every == 0:
                print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
            batch = batch.to(device, non_blocking=True)
            mini_bs = coords.shape[0]
            
            features = model(batch).cpu()
            if all_feats is None:
                all_feats = features
                all_coors = coords
            else:
                all_feats = torch.cat([all_feats, features], axis=0)
                all_coors = torch.cat([all_coors, coords], axis=0)
            
    
    print('features size: ', all_feats.shape)
    torch.save(all_feats, output_path)
    print('saved pt file: ', output_path)
    
    if save_h5_path is not None:
        asset_dict = {'features': all_feats.numpy(), 'coords': all_coors.numpy()}
        save_hdf5(save_h5_path, asset_dict, attr_dict= None, mode='w')
        print('saved h5 file: ', save_h5_path)
    
    return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--arch', type=str, default='RN50-B', help='Choose which architecture to use for extracting features.')
parser.add_argument('--ckpt_path', type=str, default=None, help='The checkpoint path for pretrained models.')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--slide_in_child_dir', default=False, action='store_true')
parser.add_argument('--sampler', default=None, type=str)
parser.add_argument('--sampler_size', default=1000, type=int)
parser.add_argument('--sampler_pool_size', default=1200, type=int)
parser.add_argument('--sampler_seed', default=42, type=int)
parser.add_argument('--color_norm', default=False, action='store_true')
parser.add_argument('--cnorm_template', default='x20-256', type=str, help='use x5-256 or x20-256 or camelyon16-x20-256')
parser.add_argument('--cnorm_method', default='macenko', type=str, help='macenko or vahadane')
parser.add_argument('--color_aug', default=None, type=str, help='Applying color augmentation to patch images.')
parser.add_argument('--patch_noise', default=None, type=str, help='Applying Guassian Blur to patch images.')
parser.add_argument('--save_h5', default=False, action='store_true')
args = parser.parse_args()


if __name__ == '__main__':

    print('initializing dataset')
    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError('No csv_path is gotten.')

    bags_dataset = Dataset_All_Bags(csv_path)
    
    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
    dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))
    if args.save_h5:
        os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)

    # sampler method
    if args.sampler is None:
        args_sampler = None
    else:
        args_sampler = {'method': args.sampler, 'size': args.sampler_size, 
            'pool_size': args.sampler_pool_size, 'seed': args.sampler_seed}

    color_normalizer = None
    if args.color_norm:
        # color normalization for pathology images
        color_normalizer = get_color_normalizer(args.cnorm_template, cn_method=args.cnorm_method)
    
    color_augmenter = None
    if args.color_aug is not None:
        # color augmentation for pathology images (patch-level)
        color_augmenter = get_color_augmenter(args.color_aug)

    print('loading model checkpoint of arch {} from {}'.format(args.arch, args.ckpt_path))
    args_pretrained = True
    if args.arch == 'RN50-B':
        model = resnet50_baseline(pretrained=True)
    elif args.arch == 'RN50-F':
        model = resnet50_full(pretrained=True)
    elif args.arch == 'RN50-MoCo-B':
        model = resnet50_MoCo('baseline', ckpt_from=args.ckpt_path) # 1024-d from layer3 of ResNet50
    elif args.arch == 'RN50-MoCo-F':
        model = resnet50_MoCo('full', ckpt_from=args.ckpt_path) # 2048-d from layer4 of ResNet50
    elif args.arch == 'RN18-SimCL':
        model = resnet18_ST(ckpt_from=args.ckpt_path) # 512-d from layer4 of ResNet18
    elif args.arch == 'ViT256-HIPT':
        model = get_vit256(ckpt_from=args.ckpt_path) # 384-d from ViT-Small
        color_normalizer = None
        args_pretrained = False
        args_sampler = None
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
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
        
        if not os.path.exists(h5_file_path):
            print('skiped slide {}, h5 file not found'.format(slide_id))
            continue
        
        slide_file_path = get_slide_fullpath(args.data_slide_dir, slide_name, 
            in_child_dir=args.slide_in_child_dir) + args.slide_ext
        print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
        print(slide_id)

        if not args.no_auto_skip and slide_id+'.pt' in dest_files:
            print('skipped {}'.format(slide_id))
            continue 

        output_pt_path = os.path.join(args.feat_dir, 'pt_files', slide_id+'.pt')
        if args.save_h5:
            output_h5_path = os.path.join(args.feat_dir, 'h5_files', slide_id+'.h5')
        else:
            output_h5_path = None
        
        time_start = time.time()
        wsi = openslide.open_slide(slide_file_path)
        output_file_path = compute_w_loader(h5_file_path, output_pt_path, wsi, 
        model = model, batch_size = args.batch_size, verbose = 1, print_every = 20, pretrained=args_pretrained,
        custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size, sampler_setting=args_sampler,                         color_normalizer=color_normalizer, color_augmenter=color_augmenter, add_patch_noise=args.patch_noise, save_h5_path=output_h5_path)
        time_elapsed = time.time() - time_start
        print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
