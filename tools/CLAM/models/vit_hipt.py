### Dependencies
# Base Dependencies
import os

# LinAlg / Stats / Plotting Dependencies
import numpy as np
from PIL import Image

# Torch Dependencies
import torch
import torchvision
from torchvision import transforms
from einops import rearrange, repeat

# Local Dependencies
import models.vision_transformer as vits


def get_vit256(ckpt_from, arch='vit_small'):
    r"""
    Builds ViT-256 Model.
    
    Args:
    - ckpt_from (str): Path to ViT-256 Model Checkpoint.
    - arch (str): Which model architecture.
    
    Returns:
    - model256 (torch.nn): Initialized model.
    """
    
    checkpoint_key = 'teacher'
    model256 = vits.__dict__[arch](patch_size=16, num_classes=0)
    for p in model256.parameters():
        p.requires_grad = False
    model256.eval()

    if os.path.isfile(ckpt_from):
        state_dict = torch.load(ckpt_from, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model256.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(ckpt_from, msg))
        
    return model256

def eval_transforms():
	"""
	"""
	mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
	eval_t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = mean, std = std)])
	return eval_t


def roll_batch2img(batch: torch.Tensor, w: int, h: int, patch_size=256):
	"""
	Rolls an image tensor batch (batch of [256 x 256] images) into a [W x H] Pil.Image object.
	
	Args:
		batch (torch.Tensor): [B x 3 x 256 x 256] image tensor batch.
		
	Return:
		Image.PIL: [W x H X 3] Image.
	"""
	batch = batch.reshape(w, h, 3, patch_size, patch_size)
	img = rearrange(batch, 'p1 p2 c w h-> c (p1 w) (p2 h)').unsqueeze(dim=0)
	return Image.fromarray(tensorbatch2im(img)[0])


def tensorbatch2im(input_image, imtype=np.uint8):
    r""""
    Converts a Tensor array into a numpy image array.
    
    Args:
        - input_image (torch.Tensor): (B, C, W, H) Torch Tensor.
        - imtype (type): the desired type of the converted numpy array
        
    Returns:
        - image_numpy (np.array): (B, W, H, C) Numpy Array.
    """
    if not isinstance(input_image, np.ndarray):
        image_numpy = input_image.cpu().float().numpy()  # convert it into a numpy array
        #if image_numpy.shape[0] == 1:  # grayscale to RGB
        #    image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)
