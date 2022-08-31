import torch
from torch import nn


class KLDivLoss(nn.Module):

	def __init__(self, opts):
		super(KLDivLoss, self).__init__()
		self.opts = opts

	def forward(self, q_mu, q_log_var, p_mu, p_var):
		feature_dim = p_mu.shape[0]

		q_var = q_log_var.exp()
		p_var_inv = 1/(p_var + 1e-8)

		log_part = torch.log(p_var).sum() - q_log_var.sum(1) #(batch,)
		k_part = feature_dim #float

		mu_sqr_part = ((q_mu-p_mu)**2 * p_var_inv).sum(1)
		trace_part = (p_var_inv * q_var).sum(1) #(batch,)

		kl_div = torch.mean(0.5*(log_part-k_part+mu_sqr_part+trace_part))
		return kl_div/feature_dim
