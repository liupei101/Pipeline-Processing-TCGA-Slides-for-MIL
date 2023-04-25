import os
import h5py
import argparse
import pandas as pd
import numpy as np
import openslide
import multiprocessing as mp
from functools import partial

from utils.utils import get_slide_id, get_slide_fullpath, calc_patches_energy
from wsi_core.wsi_utils import save_hdf5

def read_csv_pathology(path, in_child_dir=False):
	data = pd.read_csv(path)
	data['slide_name'] = data.loc[:, 'slide_id']

	data.loc[:, 'slide_id'] = data['slide_id'].apply(lambda x: get_slide_id(x, in_child_dir=in_child_dir))

	return data.loc[:, ['slide_name', 'slide_id']]

def func_main(h5_path, wsi_path, out_path, auto_skip):
	print("Processing %s" % h5_path)
	wsi = openslide.open_slide(wsi_path)

	res = None
	with h5py.File(h5_path, 'r') as f:
		if auto_skip and 'energy' in f.keys():
			print("Skiped %s" % h5_path)
			return

		coords = f['coords']
		patch_level = f['coords'].attrs['patch_level']
		patch_size = f['coords'].attrs['patch_size']
		energy = calc_patches_energy(wsi, coords, patch_level, patch_size)

		## TODO
		attr_dict = {'coords' : dict(coords.attrs), 'energy': {'func': 'utils.utils.img_energy'}}
		asset_dict = {'coords': np.array(coords), 'energy': np.array(energy)}

	save_hdf5(out_path, asset_dict, attr_dict, mode='w')


parser = argparse.ArgumentParser(description='Calculate Energy of Patches')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--out_h5_dir', type=str, default=None)
parser.add_argument('--slide_in_child_dir', default=False, action='store_true')
parser.add_argument('--auto_skip', default=False, action='store_true')
parser.add_argument('--num_workers', default=1, type=int)

args = parser.parse_args()

if __name__ == '__main__':
	print("Calculate Energy of Patches using %d workers:" % args.num_workers)
	print("\tRead data from %s" % args.data_h5_dir)
	print("\tOutput data to %s" % args.out_h5_dir)

	ref_table = read_csv_pathology(args.csv_path, args.slide_in_child_dir)
	h5_names = [_ for _ in os.listdir(args.data_h5_dir) 
						if os.path.splitext(_)[-1] == '.h5']
	slide_id = [name.split('.h5')[0] for name in h5_names]
	slide_name = [list(ref_table.loc[ref_table['slide_id'] == sid, 'slide_name'])[0]
					for sid in slide_id]
	
	h5_paths = [os.path.join(args.data_h5_dir, name) for name in h5_names]
	out_paths = [os.path.join(args.out_h5_dir, name) for name in h5_names]
	wsi_paths = [get_slide_fullpath(args.data_slide_dir, sname, args.slide_in_child_dir)
				for sname in slide_name]

	func_main_partial = partial(func_main, auto_skip=args.auto_skip)
	if args.num_workers <= 1:
		for h5_path, wsi_path, out_path in zip(h5_paths, wsi_paths, out_paths):
			func_main_partial(h5_path, wsi_path, out_path)
	else:
		with mp.Pool(processes=args.num_workers) as pool:
			pool.starmap(
				func_main_partial,
				zip(h5_paths, wsi_paths, out_paths)
			)

