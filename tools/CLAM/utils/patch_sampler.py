import numpy as np

def random_by_value(n, sample_size, values, pool_size, desc=True, random_seed=42):
	assert n == len(values)
	if pool_size < 1:
		pool_size = int(n * pool_size)

	sorted_idx = np.argsort(values)
	if desc:
		sorted_idx = sorted_idx[::-1]

	# ignore index with value less or equal to 0
	sorted_idx = [i for i in sorted_idx if values[i] > 0]

	# apply sampling
	sample_size = min(sample_size, len(sorted_idx))
	pool_size = min(pool_size, len(sorted_idx))
	assert sample_size <= pool_size

	np.random.seed(random_seed)
	pool_idx = sorted_idx[:pool_size]
	sample_idx = np.random.permutation(pool_size)[:sample_size]
	res_sample_idx = [pool_idx[_] for _ in sample_idx]

	return res_sample_idx

def random_raw(n, sample_size, random_seed=42):
	sample_size = min(sample_size, n)

	np.random.seed(random_seed)
	res_sample_idx = np.random.permutation(n)[:sample_size]

	return res_sample_idx

def sample_patches(coords, values, sampler_setting):
	met = sampler_setting['method']
	size = sampler_setting['size']
	pool_size = sampler_setting['pool_size']
	seed = sampler_setting['seed']
	n_coords = len(coords)
	if met == 'random':
		sampled_ids = random_raw(n_coords, size, seed)
		coords_res = coords[sampled_ids]
	elif met == 'random_be':
		sampled_ids = random_by_value(n_coords, size, values, 
			pool_size, random_seed=seed)
		coords_res = coords[sampled_ids]
	else:
		print("No specify sampler.")

	return coords_res
	