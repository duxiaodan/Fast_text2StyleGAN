import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import common, train_utils
from criteria import w_norm, clip_loss, KLDiv_loss
from criteria.lpips.lpips import LPIPS
from configs import data_configs
from datasets.images_dataset import ImagesDataset
from models.t2s import t2s

class Coach:
	def __init__(self, opts):
		self.opts = opts
		self.global_step = 0
		self.just_started = True
		self.device = 'cuda'  
		self.opts.device = self.device

		if self.opts.use_wandb:
			from utils.wandb_utils import WBLogger
			self.wb_logger = WBLogger(self.opts)

		# Initialize network
		self.net = t2s(self.opts).to(self.device)
		self.global_step = self.net.initial_step
		print('Initial step is', self.global_step)

		# Estimate latent_avg via dense sampling if latent_avg is not available
		if self.net.latent_avg is None:
			self.net.latent_avg = self.net.decoder.mean_latent(int(1e5))[0].detach()

		# Initialize loss
		if self.opts.lpips_lambda > 0:
			self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
		if self.opts.w_norm_lambda > 0:
			self.w_norm_loss = w_norm.WNormLoss(start_from_latent_avg=self.opts.start_from_latent_avg)
		if self.opts.clip_lambda > 0:
			self.clip_loss = clip_loss.CLIPLoss(self.opts).to(self.device).eval()
		if self.opts.KL_lambda > 0:
			if self.opts.bottleNeck_dim>0:
				self.p_var = torch.ones(self.opts.bottleNeck_dim, device=self.device)
				self.p_mu = torch.zeros(self.opts.bottleNeck_dim, device=self.device)
			elif self.opts.learn_in_z:
				self.p_var = torch.ones(512, device=self.device)
				self.p_mu = torch.zeros(512, device=self.device)
			else:
				raise Exception('If not learning in z space, bottleNeck_dim should be a positive int')
			self.kl_loss = KLDiv_loss.KLDivLoss(self.opts).to(self.device).eval()

		# Initialize optimizer
		self.optimizer = self.configure_optimizers()

		# Initialize dataset
		self.train_dataset, self.test_dataset = self.configure_datasets()
		self.train_dataloader = DataLoader(self.train_dataset,
										   batch_size=self.opts.batch_size,
										   shuffle=True,
										   num_workers=int(self.opts.workers),
										   drop_last=True)
		self.test_dataloader = DataLoader(self.test_dataset,
										  batch_size=self.opts.test_batch_size,
										  shuffle=False,
										  num_workers=int(self.opts.test_workers),
										  drop_last=True)

		# Fix some training images for logging
		self.fixed_train_dataloader = DataLoader(self.train_dataset,
										   batch_size=self.opts.batch_size,
										   shuffle=False,
										   num_workers=int(self.opts.workers),
										   drop_last=True)

		fixed_x, fixed_e = self.prepare_fixed_data(self.fixed_train_dataloader, self.opts, 10) # Default is 10 fixed images. Change it if you need
		self.fixed_x, self.fixed_e = fixed_x.to(self.device).float(), fixed_e.to(self.device).float()
		if self.opts.normalize_clip:
			self.fixed_e = self.fixed_e/(self.fixed_e.norm(dim=1,keepdim=True)+1e-8)
		# Initialize logger
		log_dir = os.path.join(self.opts.exp_dir, 'logs')
		self.log_dir = log_dir
		os.makedirs(log_dir, exist_ok=True)

		# Initialize checkpoint dir
		self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
		os.makedirs(self.checkpoint_dir, exist_ok=True)
		self.best_val_loss = None
		if self.opts.save_interval is None:
			self.opts.save_interval = self.opts.max_steps

	def train(self):
		self.net.train()
		while self.global_step < self.opts.max_steps:
			for batch_idx, batch in enumerate(self.train_dataloader):
				self.optimizer.zero_grad()
				x, e, _ = batch
				x, e = x.to(self.device).float(), e.to(self.device).float()
				if self.opts.normalize_clip:
					e = e/(e.norm(dim=1,keepdim=True)+1e-8)
				if self.opts.encoder_type == 'TwoChannelCLIPMapper':
					x_hat, mu, log_var, prior_mu, latent = self.net.forward(e, img=x, resize=not self.opts.restore_resize, return_latents=True)
				else:
					x_hat, mu, log_var, prior_mu, latent = self.net.forward(e, resize=not self.opts.restore_resize,return_latents=True)

				loss, loss_dict = self.calc_loss(x, x_hat, latent, mu, log_var, prior_mu, e)
				loss.backward()
				self.optimizer.step()

				# Logging related
				if self.global_step % self.opts.image_interval == 0 or (self.global_step < 1000 and self.global_step % 25 == 0):
					self.parse_and_log_images(x, x_hat, title='images/train/faces')
					self.val_on_fixed()
				if self.global_step % self.opts.board_interval == 0:
					self.print_metrics(loss_dict, prefix='train')
					self.log_metrics(loss_dict, prefix='train')

				# Log images of first batch to wandb
				if self.opts.use_wandb and batch_idx == 0:
					self.wb_logger.log_images_to_wandb(x, x_hat, prefix="train", step=self.global_step, opts=self.opts)

				# Validation on images other than training data 
				if not self.just_started:
					val_loss_dict = None
					if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
						val_loss_dict = self.validate()
						if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
							self.best_val_loss = val_loss_dict['loss']
							self.checkpoint_me(val_loss_dict, is_best=True)

					if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
						if val_loss_dict is not None:
							self.checkpoint_me(val_loss_dict, is_best=False)
						else:
							self.checkpoint_me(loss_dict, is_best=False)

				if self.global_step == self.opts.max_steps:
					print('OMG, finished training!')
					break

				self.global_step += 1
				# print(self.global_step)
				if self.just_started:
					self.just_started = False
	def val_on_fixed(self):
		self.net.eval()
		with torch.no_grad():
			if self.opts.encoder_type == 'TwoChannelCLIPMapper':
				x_hat, mu, log_var, prior_mu, latent = self.net.forward(self.fixed_e, img=self.fixed_x, resize=not self.opts.restore_resize, return_latents=True)
			else:
				x_hat, mu, log_var, prior_mu, latent = self.net.forward(self.fixed_e, resize=not self.opts.restore_resize,return_latents=True)
			if self.opts.sample_multi:
				x_hats = {}
				x_hats['reconstruction'] = x_hat
				for i in range(3):
					if self.opts.encoder_type == 'TwoChannelCLIPMapper':
						x_hat_random, _, _, _, _ = self.net.forward(self.fixed_e, img='random', resize=not self.opts.restore_resize, return_latents=True)
					else:
						x_hat_random, _, _, _, _ = self.net.forward(self.fixed_e, resize=not self.opts.restore_resize,return_latents=True)
					x_hats[f'random_{i}'] = x_hat_random
				self.parse_and_log_images(self.fixed_x, x_hats,
									title='images/train/fixed_faces',
									subscript='Fixed',sample_multi=True)
			else:
				self.parse_and_log_images(self.fixed_x, x_hat,
									title='images/train/fixed_faces',
									subscript='Fixed')

		# Log images of first batch to wandb
		if self.opts.use_wandb:
			self.wb_logger.log_images_to_wandb(self.fixed_x, x_hat, prefix="train_fixed", step=self.global_step, opts=self.opts)

		self.net.train()

	def validate(self):
		self.net.eval()
		agg_loss_dict = []
		for batch_idx, batch in enumerate(self.test_dataloader):
			#Only validate on the first 19 batches of the test dataloader. 
			if batch_idx >= 19:
				break
			with torch.no_grad():
				x, e, _ = batch
				x, e = x.to(self.device).float(), e.to(self.device).float()
				if self.opts.normalize_clip:
					e = e/(e.norm(dim=1,keepdim=True)+1e-8)
				if self.opts.encoder_type == 'TwoChannelCLIPMapper':
					x_hat, mu, log_var, prior_mu, latent = self.net.forward(e, img=x, resize=not self.opts.restore_resize, return_latents=True)
				else:	
					x_hat, mu, log_var, prior_mu, latent = self.net.forward(e, resize=not self.opts.restore_resize, return_latents=True)
				loss, cur_loss_dict = self.calc_loss(x, x_hat, latent, mu, log_var, prior_mu, e)
			agg_loss_dict.append(cur_loss_dict)

			# Logging related
			self.parse_and_log_images(x, x_hat,
									  title='images/test/faces',
									  subscript='{:04d}'.format(batch_idx))

			# Log images of first batch to wandb
			if self.opts.use_wandb and batch_idx == 0:
				self.wb_logger.log_images_to_wandb(x, x_hat, prefix="test", step=self.global_step, opts=self.opts)

			# For first step just do sanity test on small amount of data
			if self.global_step == 0 and batch_idx >= 4:
				self.net.train()
				return None

		loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
		self.log_metrics(loss_dict, prefix='test')
		self.print_metrics(loss_dict, prefix='test')

		self.net.train()
		return loss_dict

	def checkpoint_me(self, loss_dict, is_best):
		save_name = 'best_model.pt' if is_best else f'iteration_{self.global_step}.pt'
		save_dict = self.__get_save_dict()
		checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
		torch.save(save_dict, os.path.join(self.checkpoint_dir, 'latest_model.pt'))
		torch.save(save_dict, checkpoint_path)		
		with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
			if is_best:
				f.write(f'**Best**: Step - {self.global_step}, Loss - {self.best_val_loss} \n{loss_dict}\n')
				if self.opts.use_wandb:
					self.wb_logger.log_best_model()
			else:
				f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

	def configure_optimizers(self):
		params = list(filter(lambda p: p.requires_grad, self.net.encoder.parameters()))
		if self.opts.train_decoder:
			params += list(self.net.decoder.parameters())
		if self.opts.optim_name == 'adam':
			optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
		else:
			raise Exception("Haven't implemented this optimizer yet")
		return optimizer

	def configure_datasets(self):
		if self.opts.dataset_type not in data_configs.DATASETS.keys():
			Exception(f'{self.opts.dataset_type} is not a valid dataset_type')
		print(f'Loading dataset for {self.opts.dataset_type}')
		dataset_args = data_configs.DATASETS[self.opts.dataset_type]
		transforms_dict = dataset_args['transforms'](self.opts).get_transforms()

		train_dataset = ImagesDataset(source_root=dataset_args['train_source_root'],
									transform=transforms_dict['transform_gt_train'],
									opts=self.opts)
		test_dataset = ImagesDataset(source_root=dataset_args['test_source_root'],
									transform=transforms_dict['transform_test'],
									opts=self.opts)
		if self.opts.use_wandb:
			self.wb_logger.log_dataset_wandb(train_dataset, dataset_name="Train")
			self.wb_logger.log_dataset_wandb(test_dataset, dataset_name="Test")
		print(f"Number of training samples: {len(train_dataset)}")
		print(f"Number of test samples: {len(test_dataset)}")
		return train_dataset, test_dataset

	def calc_loss(self, x, x_hat, latent, mu, log_var, prior_mu, e):
		loss_dict = {}
		loss = 0.0

		if self.opts.lpips_lambda > 0:
			loss_lpips = self.lpips_loss(x_hat, x)
			loss_dict['loss_lpips'] = float(loss_lpips)
			loss += loss_lpips * self.opts.lpips_lambda
		if self.opts.w_norm_lambda > 0:
			loss_w_norm = self.w_norm_loss(latent, self.net.latent_avg)
			loss_dict['loss_w_norm'] = float(loss_w_norm)
			loss += loss_w_norm * self.opts.w_norm_lambda
		if self.opts.clip_lambda > 0:
			loss_clip = self.clip_loss(x_hat, e)
			loss_dict['loss_clip'] = float(loss_clip)
			loss += loss_clip * self.opts.clip_lambda
		if self.opts.KL_lambda > 0:
			assert mu is not None and log_var is not None
			if prior_mu is None:
				prior_mu = self.p_mu
			loss_kl = self.kl_loss(mu, log_var, prior_mu, self.p_var)
			loss_dict['loss_kl'] = float(loss_kl)
			loss += loss_kl * self.opts.KL_lambda

		loss_dict['loss'] = float(loss)
		return loss, loss_dict

	def log_metrics(self, metrics_dict, prefix):
		if self.opts.use_wandb:
			self.wb_logger.log(prefix, metrics_dict, self.global_step)

	def print_metrics(self, metrics_dict, prefix):
		print(f'Metrics for {prefix}, step {self.global_step}')
		for key, value in metrics_dict.items():
			print(f'\t{key} = ', value)

	def parse_and_log_images(self, x, x_hat, title, subscript=None, display_count=10, sample_multi=False):
		im_data = []
		display_count = min(display_count, x.shape[0])
		for i in range(display_count):
			if not sample_multi:
				cur_im_data = {
					'input_face': common.tensor2im(x[i]),
					'output_face': common.tensor2im(x_hat[i]),
				}
			else:
				cur_im_data = {
					'input_face': common.tensor2im(x[i]),
					'output_face': common.tensor2im(x_hat['reconstruction'][i]),
				}
				for j in range(3):
					cur_im_data[f'output_face: noise {j}'] = common.tensor2im(x_hat[f'random_{j}'][i])
			im_data.append(cur_im_data)
		self.log_images(title, im_data=im_data, subscript=subscript)

	def log_images(self, name, im_data, subscript=None, log_latest=False):
		fig = common.vis_faces2(im_data)
		step = self.global_step
		if log_latest:
			step = 0
		if subscript:
			path = os.path.join(self.log_dir, name, f'{os.path.split(self.opts.exp_dir)[-1]}_{subscript}_{step:04d}.jpg')
		else:
			path = os.path.join(self.log_dir, name, f'{os.path.split(self.opts.exp_dir)[-1]}_{step:04d}.jpg')
		os.makedirs(os.path.dirname(path), exist_ok=True)
		fig.savefig(path)
		plt.close(fig)

	def __get_save_dict(self):
		save_dict = {
			'state_dict': self.net.state_dict(),
			'opts': vars(self.opts),
			'iteration': self.global_step
		}
		# save the latent avg in state_dict for inference if truncation of w was used during training
		if self.opts.start_from_latent_avg:
			save_dict['latent_avg'] = self.net.latent_avg
		return save_dict

	def prepare_fixed_data(self, loader, opts, num=10):
		iterator = iter(loader)
		x, e, i = iterator.next()
		return x[:num], e[:num]
