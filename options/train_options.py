from argparse import ArgumentParser
from configs.paths_config import model_paths


class TrainOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		self.parser.add_argument('--exp_dir', default='./logs/my_training', type=str, help='Path to experiment output directory')
		self.parser.add_argument('--sequential', default=False, action="store_true", help='Running sequential jobs on SLURM?')
		self.parser.add_argument('--dataset_type', default='ffhq_encode', type=str, help='Type of dataset/experiment to run')
		self.parser.add_argument('--encoder_type', default='TwoChannelCLIPMapper', type=str, help='Which encoder to use')
		self.parser.add_argument('--mapper_type', default='cVAEMapper', type=str, help='Which mapper to use')
		self.parser.add_argument('--input_nc', default=3, type=int, help='Number of input image channels to the t2s encoder')
		self.parser.add_argument('--label_nc', default=0, type=int, help='Number of input label channels to the t2s encoder')
		self.parser.add_argument('--output_size', default=1024, type=int, help='Output size of generator')
		self.parser.add_argument('--restore_resize', default=False, action="store_true", help='Outputing original size results for loss computation?')
		self.parser.add_argument('--bottleNeck_dim', default=128, type=int, help='For two channel vae method only: the output dim of vae encoder.')
		self.parser.add_argument('--truncation', default=1.0, type=float, help='truncation for training')
		self.parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
		self.parser.add_argument('--test_batch_size', default=10, type=int, help='Batch size for testing and inference')
		self.parser.add_argument('--workers', default=8, type=int, help='Number of train dataloader workers')
		self.parser.add_argument('--test_workers', default=8, type=int, help='Number of test/inference dataloader workers')
		self.parser.add_argument('--learning_rate', default=0.0002, type=float, help='Optimizer learning rate')
		self.parser.add_argument('--optim_name', default='adam', type=str, help='Which optimizer to use')
		self.parser.add_argument('--train_decoder', default=False, type=bool, help='Whether to train the decoder model')
		self.parser.add_argument('--start_from_latent_avg', action='store_true', help='Whether to add average latent vector to generate codes from encoder.')
		self.parser.add_argument('--learn_in_w', action='store_true', help='Whether to learn in w space instead of w+')
		self.parser.add_argument('--learn_in_z', action='store_true', help='Whether to learn in z space instead of w+')
		self.parser.add_argument('--lr_mul', default=1.0, type=float, help='learning rate multiplier')
		self.parser.add_argument('--normalize_clip', action='store_true', help='Whether to normalize clip embeddings')

		self.parser.add_argument('--lpips_lambda', default=1.0, type=float, help='LPIPS loss multiplier factor')
		self.parser.add_argument('--w_norm_lambda', default=0.0002, type=float, help='W-norm loss multiplier factor')
		self.parser.add_argument('--clip_lambda', default=1.0, type=float, help='CLIP-based feature similarity loss multiplier factor')
		self.parser.add_argument('--KL_lambda', default=0.2, type=float, help='KL divergence loss multiplier factor')
		
		self.parser.add_argument('--stylegan_weights', default=model_paths['stylegan_ffhq'], type=str, help='Path to StyleGAN model weights')
		self.parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to encoder model checkpoint; or "auto" for automatic loading; or None for new trial')
		self.parser.add_argument('--max_steps', default=500000, type=int, help='Maximum number of training steps')
		self.parser.add_argument('--image_interval', default=1000, type=int, help='Interval for logging train images during training')
		self.parser.add_argument('--board_interval', default=50, type=int, help='Interval for logging metrics to tensorboard')
		self.parser.add_argument('--val_interval', default=10000, type=int, help='Validation interval')
		self.parser.add_argument('--save_interval', default=500, type=int, help='Model checkpoint interval')
		self.parser.add_argument('--sample_multi', default=False, action="store_true", help='Sample multiple output images for visualization?')
		# arguments for weights & biases support
		self.parser.add_argument('--use_wandb', action="store_true", help='Whether to use Weights & Biases to track experiment.')

	def parse(self):
		opts = self.parser.parse_args()
		return opts
