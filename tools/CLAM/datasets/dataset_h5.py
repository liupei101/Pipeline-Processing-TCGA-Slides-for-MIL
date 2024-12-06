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


func_ToPILImage = transforms.ToPILImage(mode='RGB')

def eval_transforms(*args, **kwargs):
	mean = (0.485, 0.456, 0.406)
	std = (0.229, 0.224, 0.225)
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
			self.roi_transforms = eval_transforms()
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
		imagenet_pretrained=False,
		custom_transforms=None,
		custom_downsample=1,
		target_patch_size=-1,
		sampler_setting=None,
		color_normalizer=None,
		color_augmenter=None, 
		add_patch_noise=None,
		vertical_flip=False,
		):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			imagenet_pretrained (bool): Use ImageNet transforms
			custom_transforms (callable, optional): Optional transform to be applied on a sample
			custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
			target_patch_size (int): Custom defined image size before embedding
			sampler_setting (dict): Setting for sampling patches
			color_normalizer (class): Normalize patches to the same color space
			color_augmenter (class): Augment the color space of patch images
			add_patch_noise (str): Add noise to patch images
			vertical_flip (bool): Apply vertical flip to patch images
		"""
		self.imagenet_pretrained=imagenet_pretrained
		self.wsi = wsi
		self.sampler_setting = sampler_setting
		self.color_normalizer = color_normalizer
		self.color_augmenter = color_augmenter
		self.add_patch_noise = add_patch_noise
		self.vertical_flip = vertical_flip
		self.hugging_face_format = False
        
        if custom_transforms is None:
			self.roi_transforms = eval_transforms()
		else:
			self.roi_transforms = custom_transforms
			from transformers.models.clip.processing_clip import CLIPProcessor
			if isinstance(self.roi_transforms, CLIPProcessor):
				self.hugging_face_format = True
			else:
				self.hugging_face_format = False

		self.pixel_operation = self.color_normalizer is None and self.color_augmenter is None and self.add_patch_noise is None
		
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
		print('-- imagenet_pretrained: ', self.imagenet_pretrained)
		print('-- patches sampler:', self.sampler_setting)
		print('-- color normalization:', self.color_normalizer)
		print('-- color argmentation:', self.color_augmenter)
		print('-- add_patch_noise:', self.add_patch_noise)
		print('-- vertical_flip:', self.vertical_flip)
		print('-- transformations: ', self.roi_transforms)

	def __getitem__(self, idx):
		coord = self.coords[idx]
		img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
		
		# Resize image first according to the specified target patch size
		if self.target_patch_size is not None and self.target_patch_size[0] != self.patch_size:
			img = img.resize(self.target_patch_size)

		if self.vertical_flip is True:
			img = img.transpose(Image.FLIP_TOP_BOTTOM)

		if self.pixel_operation:
			# cast as np ndarray for the operations on image pixels
			img = np.asarray(img).astype(np.uint8)
			
			if self.color_normalizer is not None:
				img = color_normalization(img, self.color_normalizer)
			elif self.color_augmenter is not None:
				img = color_augmentation(img, self.color_augmenter)
			else:
				pass

			if self.add_patch_noise is not None:
				img = patch_gaussian_blur(img, self.add_patch_noise)
			
			# NOTE: restore PIL.Image for running custom transforms
			img = func_ToPILImage(img)
		
		if self.hugging_face_format:
			res = self.roi_transforms(images=img, return_tensors="pt")
			img = res['pixel_values'][0].unsqueeze(0)
		else:
			img = self.roi_transforms(img).unsqueeze(0)

		return img, coord


class Dataset_All_Bags(Dataset):

	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]




