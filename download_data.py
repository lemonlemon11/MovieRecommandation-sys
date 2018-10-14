import os
from urllib.request import urlretrieve
from tqdm import tqdm


def dowload_data():
	"""
	download the data
	Options: ml-1m, ml-10m, ml-latest
	"""
	data_name = 'ml-1m'
	save_path = 'movielens/'
	dowload_path = save_path + data_name + '.zip'
	url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	if os.path.exists(dowload_path):
		print(f'{data_name} already exists')
	else:
		with DLProgress(unit='B', unit_scale=True, miniters=1, desc=f'Downloading {data_name}') as pbar:
			urlretrieve(url, dowload_path, pbar.hook)


class DLProgress(tqdm):
	last_block = 0

	def hook(self, block_num=1, block_size=1, total_size=None):
		"""
		a hook function
		"""
		self.total=total_size
		self.update((block_num - self.last_block) * block_size)
		self.last_block = block_num


dowload_data()
