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
from torchvision import transforms
from models.resnet_custom import resnet50_baseline, resnet50_full, resnet50_MoCo, resnet18_ST
import argparse
from utils.utils import print_network, collate_features
from utils.utils import get_slide_id, get_slide_fullpath
from utils.utils import get_color_normalizer, get_color_augmenter
from utils.file_utils import save_hdf5
from PIL import Image
import h5py
import openslide
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def compute_w_loader(arch, file_path, output_path, wsi, model,
    batch_size = 8, verbose = 0, print_every=20, imagenet_pretrained=True, 
    custom_downsample=1, target_patch_size=-1, sampler_setting=None, custom_transforms=None,
    color_normalizer=None, color_augmenter=None, add_patch_noise=None, vertical_flip=False, 
    save_h5_path=None, **kws):
    """
    args:
        arch: the name of model to use
        file_path: directory of bag (.h5 file)
        output_path: directory to save computed features (.h5 file)
        model: pytorch model
        batch_size: batch_size for computing features in batches
        verbose: level of feedback
        imagenet_pretrained: use weights pretrained on imagenet
        custom_downsample: custom defined downscale factor of image patches
        target_patch_size: custom defined, rescaled image size before embedding
        sampler_setting: custom defined, samlping settings
        custom_transforms: custom defined, used to transform images, e.g., mean and std normalization.
        color_normalizer: normalization for color space of pathology images
        color_augmenter: color augmentation for patch images
        add_patch_noise: adding noise to patch images
        save_h5_path: path to save features as h5 files
    """
    dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, imagenet_pretrained=imagenet_pretrained, 
        custom_downsample=custom_downsample, target_patch_size=target_patch_size, 
        sampler_setting=sampler_setting, color_normalizer=color_normalizer, 
        color_augmenter=color_augmenter, add_patch_noise=add_patch_noise, 
        vertical_flip=vertical_flip, custom_transforms=custom_transforms)
    kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}   #num_workers 4->0
    loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

    if verbose > 0:
        print('processing {}: total of {} batches'.format(file_path,len(loader)))

    proj_to_contrast = None if 'proj_to_contrast' not in kws else kws['proj_to_contrast']
    if proj_to_contrast is not None:
        if proj_to_contrast == 'N':
            print("[warning] Image features will not be projected into VL contrastive space.")
        elif proj_to_contrast == 'Y':
            print("[info] Image features will be projected into VL contrastive space.")
        elif proj_to_contrast in ['NY', 'YN']:
            print("[info] Save both the image features projected and not projected into VL contrastive space.")
            assert isinstance(output_path, tuple), f"Two output paths are expected for proj_to_contrast = {proj_to_contrast}."
            if save_h5_path is not None:
                assert isinstance(save_h5_path, tuple), f"Two save_h5 paths are expected for proj_to_contrast = {proj_to_contrast}."            
        else:
            raise ValueError(f"Invalid value of `proj_to_contrast` ({proj_to_contrast}).")

    all_feats = None
    all_coors = None
    for count, (batch, coords) in enumerate(loader):
        coords = torch.from_numpy(coords)
        with torch.no_grad():   
            if count % print_every == 0:
                print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
            batch = batch.to(device, non_blocking=True)
            mini_bs = coords.shape[0]
            
            if arch == 'CONCH':
                features = conch_encoder_image(model, batch, proj_to_contrast)
            elif arch == 'CLIP' or arch == 'PLIP':
                features = hf_clip_encoder_image(model, batch, proj_to_contrast)
            elif arch == 'OGCLIP':
                features = clip_encoder_image(model, batch, proj_to_contrast)
            else:
                features = model(batch)

            features = features.cpu() if not isinstance(features, tuple) else (features[0].cpu(), features[1].cpu())
            
            if all_feats is None:
                all_feats = features
                all_coors = coords
            else:
                if isinstance(all_feats, tuple) and isinstance(features, tuple):
                    all_feats = (torch.cat([all_feats[0], features[0]], axis=0), torch.cat([all_feats[1], features[1]], axis=0))
                else:
                    all_feats = torch.cat([all_feats, features], axis=0)

                all_coors = torch.cat([all_coors, coords], axis=0)
    
    if isinstance(all_feats, tuple):
        print("two features' size:", all_feats[0].shape)
        torch.save(all_feats[0], output_path[0])
        torch.save(all_feats[1], output_path[1])
        print("saved pt files:", output_path)
    else:
        print('features size:', all_feats.shape)
        torch.save(all_feats, output_path)
        print('saved pt file:', output_path)
    
    if save_h5_path is not None:
        if isinstance(all_feats, tuple):
            asset_dict_0 = {'features': all_feats[0].numpy(), 'coords': all_coors.numpy()}
            save_hdf5(save_h5_path[0], asset_dict_0, attr_dict=None, mode='w')
            asset_dict_1 = {'features': all_feats[1].numpy(), 'coords': all_coors.numpy()}
            save_hdf5(save_h5_path[1], asset_dict_1, attr_dict=None, mode='w')
            print('saved h5 file:', save_h5_path)
        else:
            asset_dict = {'features': all_feats.numpy(), 'coords': all_coors.numpy()}
            save_hdf5(save_h5_path, asset_dict, attr_dict=None, mode='w')
            print('saved h5 file:', save_h5_path)
    
    return output_path

@torch.no_grad()
def conch_encoder_image(conch_model, batch, proj_contrast='Y'):
    # Use CONCH's built-in functions
    vis_features = conch_model.visual.forward_no_head(batch, normalize=False)
    
    if proj_contrast == 'N':
        image_features = vis_features

    elif proj_contrast == 'Y':
        image_features = conch_model.visual.forward_project(vis_features)

    elif proj_contrast in ['NY', 'YN']:
        image_features = (vis_features, conch_model.visual.forward_project(vis_features))

    return image_features

@torch.no_grad()
def hf_clip_encoder_image(hf_clip, batch, proj_contrast='Y'):
    """
    This follows the implementation of HuggingFace - transformers
    """
    # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
    output_attentions = hf_clip.config.output_attentions
    output_hidden_states = (hf_clip.config.output_hidden_states)
    return_dict = hf_clip.config.use_return_dict

    vision_outputs = hf_clip.vision_model(
        pixel_values=batch,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    pooled_output = vision_outputs[1]  # pooled_output
    
    if proj_contrast == 'N':
        image_features = pooled_output

    elif proj_contrast == 'Y':
        image_features = hf_clip.visual_projection(pooled_output)

    elif proj_contrast in ['NY', 'YN']:
        image_features = (pooled_output, hf_clip.visual_projection(pooled_output))

    return image_features

@torch.no_grad()
def clip_encoder_image(clip_model, batch, proj_contrast='Y'):
    # Use CLIP's built-in functions
    vis_features = clip_model.encode_image(
        batch, 
        proj_contrast=False
    )
    
    if proj_contrast == 'N':
        image_features = vis_features

    elif proj_contrast == 'Y':
        image_features = vis_features @ clip_model.visual.proj

    elif proj_contrast in ['NY', 'YN']:
        image_features = (vis_features, vis_features @ clip_model.visual.proj)

    return image_features

parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--arch', type=str, default='CONCH', choices=['RN50-B', 'RN50-F', 'RN18-SimCL', 'ViT256-HIPT', 'CTransPath', 'OGCLIP', 'CLIP', 'PLIP', 'CONCH', 'UNI'], help='Choose which architecture to use for extracting features.')
parser.add_argument('--ckpt_path', type=str, default=None, help='The checkpoint path for pretrained models.')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--feat_dir_ext', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=256)
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
parser.add_argument('--vertical_flip', default=False, action='store_true', help='Applying Vertical Flip to patch images.')
parser.add_argument('--save_h5', default=False, action='store_true')
parser.add_argument('--proj_to_contrast', type=str, default='Y', choices=['Y', 'N', 'YN', 'NY'], help='If projecting image features into VL contrast space.')
parser.add_argument('--clip_type', default='ViT-B/32', type=str, help='used for specifying the CLIP model.')
args = parser.parse_args()


if __name__ == '__main__':

    print('initializing dataset')
    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError('No csv_path is gotten.')

    bags_dataset = Dataset_All_Bags(csv_path)

    # prepare directories to save patch features
    args_proj_to_contrast = args.proj_to_contrast
    if args_proj_to_contrast in ['YN', 'NY']:
        assert args.feat_dir_ext is not None, f"Got proj_to_contrast ({args_proj_to_contrast}); please specify an extra directory to save features."
        args_feat_dir = (args.feat_dir, args.feat_dir_ext)
        for feat_dir in args_feat_dir:
            os.makedirs(feat_dir, exist_ok=True)
            os.makedirs(os.path.join(feat_dir, 'pt_files'), exist_ok=True)
            if args.save_h5:
                os.makedirs(os.path.join(feat_dir, 'h5_files'), exist_ok=True)
        print(f"Dirs to save raw / projected feats: {args_feat_dir[0]} / {args_feat_dir[1]}.")
    else:
        args_feat_dir = args.feat_dir
        os.makedirs(args_feat_dir, exist_ok=True)
        os.makedirs(os.path.join(args_feat_dir, 'pt_files'), exist_ok=True)
        if args.save_h5:
            os.makedirs(os.path.join(args_feat_dir, 'h5_files'), exist_ok=True)

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
    args_imagenet_pretrained = True
    args_custom_transforms = None
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
        color_normalizer = None
        args_imagenet_pretrained = False
        args_sampler = None
        print(f"[warning] Due to the use of {args.arch}, your color_normalizer and patch sampler are forced to be None, not active.")
    elif args.arch == 'ViT256-HIPT':
        from models.vit_hipt import get_vit256
        model = get_vit256(ckpt_from=args.ckpt_path) # 384-d from ViT-Small
        color_normalizer = None
        args_imagenet_pretrained = False
        args_sampler = None
        print(f"[warning] Due to the use of {args.arch}, your color_normalizer and patch sampler are forced to be None, not active.")
    elif args.arch == 'CTransPath':
        from models.ctran import ctranspath
        model = ctranspath(ckpt_from=args.ckpt_path) # 768-d from CTransPath
        color_normalizer = None
        args_imagenet_pretrained = False
        args_sampler = None
        args_custom_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        ])
        print(f"[warning] Due to the use of {args.arch}, only using custom transforms and all other arguments are not active.")
    elif args.arch == 'OGCLIP':
        from models import clip
        model, preprocess = clip.load(args.clip_type, device=device, download_root=args.ckpt_path) # "ViT-B/32"
        color_normalizer = None
        args_imagenet_pretrained = False
        args_sampler = None
        args_custom_transforms = preprocess
        print(f"[warning] Due to the use of {args.arch}-{args.clip_type}, only using custom transforms and all other arguments are not active.")
    elif args.arch == 'CLIP':
        from transformers import CLIPProcessor, CLIPModel
        model = CLIPModel.from_pretrained(args.ckpt_path)
        processor = CLIPProcessor.from_pretrained(args.ckpt_path)
        color_normalizer = None
        args_imagenet_pretrained = False
        args_sampler = None
        args_custom_transforms = processor
        print(f"[warning] Due to the use of {args.arch}, only using custom transforms and all other arguments are not active.")
    elif args.arch == 'PLIP':
        from transformers import CLIPProcessor, CLIPModel
        model = CLIPModel.from_pretrained(args.ckpt_path)
        processor = CLIPProcessor.from_pretrained(args.ckpt_path)
        color_normalizer = None
        args_imagenet_pretrained = False
        args_sampler = None
        args_custom_transforms = processor
        print(f"[warning] Due to the use of {args.arch}, only using custom transforms and all other arguments are not active.")
    elif args.arch == 'CONCH':
        from models.conch import create_model_from_pretrained
        model, preprocess = create_model_from_pretrained(
            "conch_ViT-B-16", 
            checkpoint_path=args.ckpt_path,
            force_image_size=args.target_patch_size,
        )
        color_normalizer = None
        args_imagenet_pretrained = False
        args_sampler = None
        args_custom_transforms = preprocess
        print(f"[warning] Due to the use of {args.arch}, only using custom transforms and all other arguments are not active.")
    elif args.arch == 'UNI':
        from models.uni import get_encoder
        model, preprocess = get_encoder(enc_name='uni', assets_dir=osp.dirname(osp.dirname(args.ckpt_path)))
        color_normalizer = None
        args_imagenet_pretrained = False
        args_sampler = None
        args_enable_direct_transforms = True
        args_custom_transforms = preprocess
        print(f"[warning] Due to the use of {args.arch}, only using custom transforms and all other arguments are not active.")
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
        bag_name = slide_id + '.h5'
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
        
        if not os.path.exists(h5_file_path):
            print('skiped slide {}, h5 file not found'.format(slide_id))
            continue
        
        slide_file_path = get_slide_fullpath(
            args.data_slide_dir, slide_name, 
            in_child_dir=args.slide_in_child_dir
        ) + args.slide_ext
        print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
        print(slide_id)

        # prepare save paths 
        if isinstance(args_feat_dir, tuple):
            output_pt_path = (os.path.join(args_feat_dir[0], 'pt_files', slide_id + '.pt'), os.path.join(args_feat_dir[1], 'pt_files', slide_id + '.pt'))
            if args.save_h5:
                output_h5_path = (os.path.join(args_feat_dir[0], 'h5_files', slide_id + '.h5'), os.path.join(args_feat_dir[1], 'h5_files', slide_id + '.h5'))
            else:
                output_h5_path = None

            if args.auto_skip and os.path.exists(output_pt_path[0]) and os.path.exists(output_pt_path[1]):
                print('skipped {}'.format(slide_id))
                continue

        else:
            output_pt_path = os.path.join(args_feat_dir, 'pt_files', slide_id+'.pt')
            if args.save_h5:
                output_h5_path = os.path.join(args_feat_dir, 'h5_files', slide_id+'.h5')
            else:
                output_h5_path = None

            if args.auto_skip and os.path.exists(output_pt_path):
                print('skipped {}'.format(slide_id))
                continue
        
        time_start = time.time()
        wsi = openslide.open_slide(slide_file_path)
        output_file_path = compute_w_loader(args.arch, h5_file_path, output_pt_path, wsi, 
            model = model, batch_size = args.batch_size, verbose = 1, print_every = 20, imagenet_pretrained=args_imagenet_pretrained,
            custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size, sampler_setting=args_sampler,
            custom_transforms=args_custom_transforms, color_normalizer=color_normalizer, color_augmenter=color_augmenter,
            add_patch_noise=args.patch_noise, vertical_flip=args.vertical_flip, 
            save_h5_path=output_h5_path, proj_to_contrast=args_proj_to_contrast
        )
        time_elapsed = time.time() - time_start
        print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
