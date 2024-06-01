# internal imports
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords
from wsi_core.batch_process_utils import initialize_df
from utils.utils import get_slide_id, get_slide_fullpath, read_slides_name
# other imports
import os
import numpy as np
import time
import argparse
import pdb
import pandas as pd

EPS = 1e-5

def stitching(file_path, wsi_object, downscale = 64):
	start = time.time()
	heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(0,0,0), alpha=-1, draw_grid=False)
	total_time = time.time() - start
	
	return heatmap, total_time

def segment(WSI_object, seg_params, filter_params):
	### Start Seg Timer
	start_time = time.time()

	# Segment
	WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

	### Stop Seg Timers
	seg_time_elapsed = time.time() - start_time   
	return WSI_object, seg_time_elapsed

def patching(WSI_object, **kwargs):
	### Start Patch Timer
	start_time = time.time()

	# Patch
	file_path = WSI_object.process_contours(**kwargs)


	### Stop Patch Timer
	patch_time_elapsed = time.time() - start_time
	return file_path, patch_time_elapsed

def seg_and_patch(source, save_dir, patch_save_dir, mask_save_dir, stitch_save_dir, 
				  patch_size = 256, step_size = 256, 
				  seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				  'keep_ids': 'none', 'exclude_ids': 'none'},
				  filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}, 
				  vis_params = {'vis_level': -1, 'line_thickness': 500},
				  patch_params = {'use_padding': True, 'contour_fn': 'four_pt'},
				  patch_magnification = 20,
				  use_default_params = False, 
				  seg = False, save_mask = True, 
				  stitch= False, 
				  patch = False, auto_skip=True, process_list = None,
				  in_child_dir = False):
	

	slides = read_slides_name(source, in_child_dir)
	
	if process_list is None:
		df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)
	else:
		df = pd.read_csv(process_list)
		df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)

	mask = df['process'] == 1
	process_stack = df[mask]

	total = len(process_stack)

	legacy_support = 'a' in df.keys()
	if legacy_support:
		print('[info] detected legacy segmentation csv file, legacy support enabled')
		df = df.assign(**{'a_t': np.full((len(df)), int(filter_params['a_t']), dtype=np.uint32),
		'a_h': np.full((len(df)), int(filter_params['a_h']), dtype=np.uint32),
		'max_n_holes': np.full((len(df)), int(filter_params['max_n_holes']), dtype=np.uint32),
		'line_thickness': np.full((len(df)), int(vis_params['line_thickness']), dtype=np.uint32),
		'contour_fn': np.full((len(df)), patch_params['contour_fn'])})

	# NEW: record more information 
    # magnification: the highest magnification of the slide (level=0)
    # seg_ref_size: the default ref patch size (level=0) used in segmentation
    # patch_ref_size: the ref patch size (level=0) used in patching
    # patch_ref_step: the ref patch size (level=0) used in patching
    for new_k in ['magnification', 'seg_ref_size', 'patch_ref_size', 'patch_ref_step']:
        if new_k not in df.columns:
            df.loc[:, new_k] = -1

	seg_times = 0.
	patch_times = 0.
	stitch_times = 0.

	for i in range(total):
		df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
		idx = process_stack.index[i]
		slide = process_stack.loc[idx, 'slide_id']
		print("\n\nprogress: {:.2f}, {}/{}".format(i/total, i, total))
		print('processing {}'.format(slide))
		
		df.loc[idx, 'process'] = 0
		slide_id = get_slide_id(slide, in_child_dir=in_child_dir)

		if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
			print('{} already exist in destination location, skipped'.format(slide_id))
			df.loc[idx, 'status'] = 'already_exist'
			continue

		# Inialize WSI
		full_path = get_slide_fullpath(source, slide, in_child_dir)
		WSI_object = WholeSlideImage(full_path)
		mag = WSI_object.magnification

		# NEW: update slide DataFrame
		df.loc[idx, 'magnification'] = mag
		if mag != -1:
			downsample_level = 40 // mag
			df.loc[idx, 'seg_ref_size'] = 512 // downsample_level

			patch_downsample_level = mag / patch_magnification
			df.loc[idx, 'patch_ref_size'] = int(patch_downsample_level * patch_size + EPS)
			df.loc[idx, 'patch_ref_step'] = int(patch_downsample_level * step_size + EPS)
		else:
			# by default at 40x if mag is N/A
			df.loc[idx, 'seg_ref_size'] = 512
			patch_downsample_level = 40 / patch_magnification
			df.loc[idx, 'patch_ref_size'] = int(patch_downsample_level * patch_size + EPS)
			df.loc[idx, 'patch_ref_step'] = int(patch_downsample_level * step_size + EPS)

		if use_default_params:
			current_vis_params = vis_params.copy()
			current_filter_params = filter_params.copy()
			current_seg_params = seg_params.copy()
			current_patch_params = patch_params.copy()
			
		else:
			current_vis_params = {}
			current_filter_params = {}
			current_seg_params = {}
			current_patch_params = {}


			for key in vis_params.keys():
				if legacy_support and key == 'vis_level':
					df.loc[idx, key] = -1
				current_vis_params.update({key: df.loc[idx, key]})

			for key in filter_params.keys():
				if legacy_support and key == 'a_t':
					old_area = df.loc[idx, 'a']
					seg_level = df.loc[idx, 'seg_level']
					scale = WSI_object.level_downsamples[seg_level]
					adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
					current_filter_params.update({key: adjusted_area})
					df.loc[idx, key] = adjusted_area
				current_filter_params.update({key: df.loc[idx, key]})

			for key in seg_params.keys():
				if legacy_support and key == 'seg_level':
					df.loc[idx, key] = -1
				current_seg_params.update({key: df.loc[idx, key]})

			for key in patch_params.keys():
				current_patch_params.update({key: df.loc[idx, key]})

		if current_vis_params['vis_level'] < 0:
			if len(WSI_object.level_dim) == 1:
				current_vis_params['vis_level'] = 0
			else:	
				wsi = WSI_object.getOpenSlide()
				best_level = wsi.get_best_level_for_downsample(64)
				current_vis_params['vis_level'] = best_level

		if current_seg_params['seg_level'] < 0:
			if len(WSI_object.level_dim) == 1:
				current_seg_params['seg_level'] = 0
			else:
				wsi = WSI_object.getOpenSlide()
				best_level = wsi.get_best_level_for_downsample(64)
				current_seg_params['seg_level'] = best_level

		keep_ids = str(current_seg_params['keep_ids'])
		if keep_ids != 'none' and len(keep_ids) > 0:
			str_ids = current_seg_params['keep_ids']
			current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['keep_ids'] = []

		exclude_ids = str(current_seg_params['exclude_ids'])
		if exclude_ids != 'none' and len(exclude_ids) > 0:
			str_ids = current_seg_params['exclude_ids']
			current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['exclude_ids'] = []

		w, h = WSI_object.level_dim[current_seg_params['seg_level']] 
		if w * h > 1e8:
			print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
			df.loc[idx, 'status'] = 'failed_seg'
			continue

		df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
		df.loc[idx, 'seg_level'] = current_seg_params['seg_level']

		# NEW: update parameters for segmentation
		current_seg_params['ref_patch_size'] = df.loc[idx, 'seg_ref_size']
		if current_seg_params['ref_patch_size'] != 512:
			print(f"[info] `ref_patch_size`, 512 by default for seg, is {df.loc[idx, 'seg_ref_size']} as magnification = {mag}x.")

		seg_time_elapsed = -1
		if seg:
			WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params) 

		if save_mask:
			mask = WSI_object.visWSI(**current_vis_params)
			mask_path = os.path.join(mask_save_dir, slide_id+'.jpg')
			mask.save(mask_path)

		# NEW: update parameters for patching
		cur_patch_size = df.loc[idx, 'patch_ref_size']
		cur_step_size = df.loc[idx, 'patch_ref_step']
		if cur_patch_size != patch_size:
			print(f"[info] `patch_ref_size` is {cur_patch_size} at {mag}x when patching with {patch_size} at {patch_magnification}x.")

		patch_time_elapsed = -1 # Default time
		if patch:
			current_patch_params.update({'patch_level': 0, 'patch_size': cur_patch_size, 'step_size': cur_step_size, 
										 'save_path': patch_save_dir})
			file_path, patch_time_elapsed = patching(WSI_object = WSI_object,  **current_patch_params,)
		
		stitch_time_elapsed = -1
		if stitch:
			file_path = os.path.join(patch_save_dir, slide_id+'.h5')
			if os.path.isfile(file_path):
				heatmap, stitch_time_elapsed = stitching(file_path, WSI_object, downscale=64)
				stitch_path = os.path.join(stitch_save_dir, slide_id+'.jpg')
				heatmap.save(stitch_path)

		print("segmentation took {} seconds".format(seg_time_elapsed))
		print("patching took {} seconds".format(patch_time_elapsed))
		print("stitching took {} seconds".format(stitch_time_elapsed))
		df.loc[idx, 'status'] = 'processed'

		seg_times += seg_time_elapsed
		patch_times += patch_time_elapsed
		stitch_times += stitch_time_elapsed

	seg_times /= total
	patch_times /= total
	stitch_times /= total

	df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
	print("average segmentation time in s per slide: {}".format(seg_times))
	print("average patching time in s per slide: {}".format(patch_times))
	print("average stiching time in s per slide: {}".format(stitch_times))
		
	return seg_times, patch_times

parser = argparse.ArgumentParser(description='seg and patch')
parser.add_argument('--source', type = str,
					help='path to folder containing raw wsi image files')
parser.add_argument('--step_size', type = int, default=256,
					help='step_size')
parser.add_argument('--patch_size', type = int, default=256,
					help='patch_size')
parser.add_argument('--patch', default=False, action='store_true')
parser.add_argument('--seg', default=False, action='store_true')
parser.add_argument('--stitch', default=False, action='store_true')
parser.add_argument('--save_mask', default=False, action='store_true')
parser.add_argument('--auto_skip', default=False, action='store_true')
parser.add_argument('--save_dir', type = str,
					help='directory to save processed data')
parser.add_argument('--preset', default=None, type=str,
					help='predefined profile of default segmentation and filter parameters (.csv)')
parser.add_argument('--patch_magnification', type=int, default=20, 
					help='magnification at which to patch')
parser.add_argument('--process_list',  type = str, default=None,
					help='name of list of images to process with parameters (.csv)')
parser.add_argument('--in_child_dir',  default=False, action='store_true')

"""
Example for use:
	CUDA_VISIBLE_DEVICES=0 python3 create_patches_fp.py \
		--source ${DIR_DATA} \
		--save_dir ${DIR_SAVE_DATA}/tiles-l${LEVEL}-s${SIZE} \
		--patch_size ${SIZE} \
		--step_size ${SIZE} \
		--preset tcga.csv \
		--patch_magnification 20 \
		--seg --patch \
		--auto_skip --in_child_dir
"""

if __name__ == '__main__':
	args = parser.parse_args()

	patch_save_dir = os.path.join(args.save_dir, 'patches')
	mask_save_dir = os.path.join(args.save_dir, 'masks')
	stitch_save_dir = os.path.join(args.save_dir, 'stitches')

	if args.process_list:
		process_list = os.path.join(args.save_dir, args.process_list)
	else:
		process_list = None

	print("[info] Default path setup:")
	print('\tsource: ', args.source)
	print('\tpatch_save_dir: ', patch_save_dir)
	print('\tmask_save_dir: ', mask_save_dir)
	print('\tstitch_save_dir: ', stitch_save_dir)
	
	directories = {'source': args.source, 
				   'save_dir': args.save_dir,
				   'patch_save_dir': patch_save_dir, 
				   'mask_save_dir' : mask_save_dir, 
				   'stitch_save_dir': stitch_save_dir} 

	os.makedirs(args.save_dir, exist_ok=True)
	if args.save_mask:
		os.makedirs(mask_save_dir, exist_ok=True)
	if args.patch:
		os.makedirs(patch_save_dir, exist_ok=True)
	if args.stitch:
		os.makedirs(stitch_save_dir, exist_ok=True)

	# NOTE: we use TCGA preset parameters by default 
	# it is kept the same as `https://github.com/mahmoodlab/CLAM/blob/master/presets/tcga.csv`
	seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				  'keep_ids': 'none', 'exclude_ids': 'none'}
	filter_params = {'a_t':16, 'a_h': 4, 'max_n_holes':8}
	vis_params = {'vis_level': -1, 'line_thickness': 100}
	patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

	if args.preset:
		preset_df = pd.read_csv(os.path.join('presets', args.preset))
		for key in seg_params.keys():
			seg_params[key] = preset_df.loc[0, key]

		for key in filter_params.keys():
			filter_params[key] = preset_df.loc[0, key]

		for key in vis_params.keys():
			vis_params[key] = preset_df.loc[0, key]

		for key in patch_params.keys():
			patch_params[key] = preset_df.loc[0, key]
	
	parameters = {'seg_params': seg_params,
				  'filter_params': filter_params,
	 			  'patch_params': patch_params,
				  'vis_params': vis_params}

	print("[info] Parameters to be used for creating patches:", parameters)

	seg_times, patch_times = seg_and_patch(**directories, **parameters,
											patch_size=args.patch_size, step_size=args.step_size, 
											seg=args.seg, use_default_params=False, save_mask = args.save_mask, 
											stitch=args.stitch,
											patch_magnification=args.patch_magnification, patch=args.patch,
											process_list=process_list, auto_skip=args.auto_skip, in_child_dir=args.in_child_dir)
