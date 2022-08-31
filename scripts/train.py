"""
This file runs the main training/val loop
"""
import os
import json
import sys
import pprint
import torch
sys.path.append(".")
sys.path.append("..")

from options.train_options import TrainOptions
from training.coach import Coach


def main():
	opts = TrainOptions().parse()
	if os.path.exists(opts.exp_dir) and opts.sequential==False:
		raise Exception('Oops... {} already exists'.format(opts.exp_dir))
	if opts.learn_in_w and opts.learn_in_z:
		raise Exception('Cannot learn in w and z simultaniously.')
	if opts.start_from_latent_avg and opts.learn_in_z:
		raise Exception('Cannot learn in z while start from average w.')
	os.makedirs(opts.exp_dir,exist_ok=True)

	opts_dict = vars(opts)
	pprint.pprint(opts_dict)
	with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
		json.dump(opts_dict, f, indent=4, sort_keys=True)
	with torch.autograd.set_detect_anomaly(True):
		coach = Coach(opts)
		coach.train()


if __name__ == '__main__':
	main()
