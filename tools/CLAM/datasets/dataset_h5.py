from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle

from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils, models
import torch.nn.functional as F

from PIL import Image
import h5py
from random import randrange

from utils.patch_sampler import sample_patches
from utils.utils import color_normalization, color_augmentation, patch_gaussian_blur

# only used after color normalization
def colnor_followed_transforms(pretrained=False):
	if pretrained:
		mean = (0.485, 0.456, 0.406)
		std = (0.229, 0.224, 0.225)

	else:
		mean = (0.5,0.5,0.5)
		std = (0.5,0.5,0.5)

	trnsfrms_val = transforms.Compose(
		[
		    transforms.Lambda(lambda x: x/255.0),
		    transforms.Normalize(mean=mean, std=std)
		]
	)

	return trnsfrms_val

def eval_transforms(pretrained=False):
	if pretrained:
		mean = (0.485, 0.456, 0.406)
		std = (0.229, 0.224, 0.225)

	else:
		mean = (0.5,0.5,0.5)
		std = (0.5,0.5,0.5)

	trnsfrms_val = transforms.Compose(
					[
					 transforms.ToTensor(),
					 transforms.Normalize(mean = mean, std = std)
					]
				)

	return trnsfrms_val

class Whole_Slide_Bag(Dataset):
	def __init__(self,
		file_path,
		pretrained=False,
		custom_transforms=None,
		target_patch_size=-1,
		):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			pretrained (bool): Use ImageNet transforms
			custom_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.pretrained=pretrained
		if target_patch_size > 0:
			self.target_patch_size = (target_patch_size, target_patch_size)
		else:
			self.target_patch_size = None

		if not custom_transforms:
			self.roi_transforms = eval_transforms(pretrained=pretrained)
		else:
			self.roi_transforms = custom_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['imgs']
			self.length = len(dset)

		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['imgs']
		for name, value in dset.attrs.items():
			print(name, value)

		print('pretrained:', self.pretrained)
		print('transformations:', self.roi_transforms)
		if self.target_patch_size is not None:
			print('target_size: ', self.target_patch_size)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			img = hdf5_file['imgs'][idx]
			coord = hdf5_file['coords'][idx]
		
		img = Image.fromarray(img)
		if self.target_patch_size is not None:
			img = img.resize(self.target_patch_size)
		img = self.roi_transforms(img).unsqueeze(0)
		return img, coord

class Whole_Slide_Bag_FP(Dataset):
	def __init__(self,
		file_path,
		wsi,
		pretrained=False,
		custom_transforms=None,
		custom_downsample=1,
		target_patch_size=-1,
		sampler_setting=None,
		color_normalizer=None,
		color_augmenter=None, 
		add_patch_noise=None
		):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			pretrained (bool): Use ImageNet transforms
			custom_transforms (callable, optional): Optional transform to be applied on a sample
			custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
			target_patch_size (int): Custom defined image size before embedding
			sampler_setting (dict): Setting for sampling patches
			color_normalizer (class): Normalize patches to the same color space
			color_augmenter (class): Augment the color space of patch images
			add_patch_noise (bool): Add noise to patch images
		"""
		self.pretrained=pretrained
		self.wsi = wsi
		self.sampler_setting = sampler_setting
		self.color_normalizer = color_normalizer
		self.color_augmenter = color_augmenter
		self.add_patch_noise = add_patch_noise
		if self.color_normalizer is not None:
			self.roi_transforms = colnor_followed_transforms(pretrained=pretrained)
		elif not custom_transforms:
			self.roi_transforms = eval_transforms(pretrained=pretrained)
		else:
			self.roi_transforms = custom_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			self.patch_level = f['coords'].attrs['patch_level']
			self.patch_size = f['coords'].attrs['patch_size']
			if target_patch_size > 0:
				self.target_patch_size = (target_patch_size, ) * 2
			elif custom_downsample > 1:
				self.target_patch_size = (self.patch_size // custom_downsample, ) * 2
			else:
				self.target_patch_size = None

			self.coords = np.array(f['coords'])
			if 'energy' in f.keys():
				self.coords_energy = np.array(f['energy'])
			else:
				self.coords_energy = None

		# apply sampling to patches
		if self.sampler_setting is not None:
			self.coords = sample_patches(self.coords, self.coords_energy, self.sampler_setting)

		self.length = len(self.coords)
		self.summary()

	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['coords']
		for name, value in dset.attrs.items():
			print(name, value)

		print('\nfeature extraction settings:')
		print('-- target patch size: ', self.target_patch_size)
		print('-- pretrained: ', self.pretrained)
		print('-- patches sampler:', self.sampler_setting)
		print('-- color normalization:', self.color_normalizer)
		print('-- color argmentation:', self.color_augmenter)
		print('-- add_patch_noise:', self.add_patch_noise)
		print('-- transformations: ', self.roi_transforms)

	def __getitem__(self, idx):
		coord = self.coords[idx]
		img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

		if self.target_patch_size is not None:
			img = img.resize(self.target_patch_size)

		if self.color_normalizer is not None:
			img = np.array(img).astype(np.uint8)
			img = color_normalization(img, self.color_normalizer)
		elif self.color_augmenter is not None:
			img = np.array(img).astype(np.uint8)
			img = color_augmentation(img, self.color_augmenter)
		else:
			img = np.array(img).astype(np.uint8)

		if self.add_patch_noise is not None:
			img = patch_gaussian_blur(img, self.add_patch_noise)

		img = self.roi_transforms(img).unsqueeze(0)

		return img, coord

class Dataset_All_Bags(Dataset):

	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]




