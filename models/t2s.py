"""
This file defines the core research contribution
"""
import matplotlib
matplotlib.use('Agg')
import math
import os
import re
import torch
from torch import nn
from models.encoders import t2s_encoders
from models.stylegan2.model import Generator
from configs.paths_config import model_paths


def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt


class t2s(nn.Module):

	def __init__(self, opts):
		super(t2s, self).__init__()
		self.initial_step = 0
		self.set_opts(opts)
		# compute number of style inputs based on the output resolution
		self.opts.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
		# Define architecture
		self.encoder = self.set_encoder('encoder')
		self.decoder = Generator(self.opts.output_size, 512, 8)
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
		# Load weights if needed
		self.load_weights()

	def set_encoder(self,cmd='encoder'):
		if cmd=='encoder':
			if self.opts.encoder_type == 'StyleCLIPMapper':
				encoder = t2s_encoders.StyleCLIPMapper(self.opts)
			elif self.opts.encoder_type == "TwoChannelCLIPMapper":
				encoder = t2s_encoders.TwoChannelCLIPMapper(self.opts)
			else:
				raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
		else:
			raise Exception('{} is not a valid encoder type'.format(cmd))
		return encoder
			
	def load_weights(self):
		# If the user provide a checkpoint file, use that checkpoint
		if self.opts.checkpoint_path is not None and self.opts.checkpoint_path != 'auto':
			print('Loading t2s from checkpoint: {}'.format(self.opts.checkpoint_path))
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			print('Checkpoint keys:',ckpt.keys())
			self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
			self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
			self.__load_latent_avg(ckpt)
			self.initial_step = int(ckpt['iteration'])
		# If the 'auto' mode is on (convinient for sequential singleton jobs), pick the latest model and start from there
		elif self.opts.checkpoint_path == 'auto' and os.path.exists(os.path.join(self.opts.exp_dir, 'checkpoints','latest_model.pt')):
			print('Loading t2s from checkpoint: {}'.format(self.opts.checkpoint_path))
			ckpt = torch.load(os.path.join(self.opts.exp_dir, 'checkpoints','latest_model.pt'), map_location='cpu')
			print('Checkpoint keys:',ckpt.keys())
			self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
			self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
			self.__load_latent_avg(ckpt)
			self.initial_step = int(ckpt['iteration'])
		# If no checkpoint file, only load stylegan generator
		else:
			print('Loading decoder weights from pretrained!')
			ckpt = torch.load(self.opts.stylegan_weights)
			self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
			if self.opts.learn_in_w or self.opts.learn_in_z:
				self.__load_latent_avg(ckpt, repeat=1)
			else:
				self.__load_latent_avg(ckpt, repeat=self.opts.n_styles)

	def forward(self, e, img=None, resize=True, input_code=False, randomize_noise=True,
	            return_latents=False, zero_var=False):
		# e here is clip embedding for our case
		if input_code:
			codes = e
		else:
			if img is None:
				codes, mu, log_var, prior_mu = self.encoder(e,zero_var=zero_var)
			else:
				codes, mu, log_var, prior_mu = self.encoder(e,img, zero_var=zero_var)
			if self.opts.start_from_latent_avg:
				if self.opts.learn_in_w:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
				else:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

		input_is_latent = not (input_code or self.opts.learn_in_z)

		images, result_latent = self.decoder([codes],
											input_is_latent=input_is_latent,
											truncation=self.opts.truncation,
											truncation_latent=self.latent_avg,
											randomize_noise=randomize_noise,
											return_latents=return_latents)
		
		if resize:
			images = self.face_pool(images)
		results=(images,)
		if not input_code:
			results += (mu, log_var, prior_mu,)
		if return_latents:
			results += (result_latent,)
		
		return results

	def set_opts(self, opts):
		self.opts = opts

	def __load_latent_avg(self, ckpt, repeat=None):
		if 'latent_avg' in ckpt:
			self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
			if repeat is not None:
				self.latent_avg = self.latent_avg.repeat(repeat, 1)
		else:
			self.latent_avg = None
