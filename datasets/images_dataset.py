from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import h5py
import os
import numpy as np
from configs.paths_config import dataset_paths


class ImagesDataset(Dataset):

	def __init__(self, source_root, opts, transform=None):
		self.source_root = source_root
		self.source_paths = sorted(data_utils.make_dataset(source_root))
		self.transform = transform
		self.opts = opts
		self.dataset = 'CelebAHQ' if 'Celeb' in self.source_root else 'FFHQ'
		if self.dataset == 'CelebAHQ':
			CLIPemb_file = h5py.File(dataset_paths['celeba_clip'], "r")
		elif self.dataset == 'FFHQ':
			CLIPemb_file = h5py.File(dataset_paths['ffhq_clip'], "r")
		else:
			raise Exception("Haven't implemented this dataset")
		self.CLIPemb = np.array(CLIPemb_file["image"])

	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		path = self.source_paths[index]
		im = Image.open(path)
		im = im.convert('RGB') if self.opts.label_nc == 0 else im.convert('L')
		if self.transform:
			im = self.transform(im)

		imgid = int(path[path.rfind('/')+1:-4])
		img_emb = self.CLIPemb[imgid]
		return im, img_emb, imgid

