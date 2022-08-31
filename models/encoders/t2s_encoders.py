import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from models.encoders import latent_mappers
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module

class StyleCLIPMapper(nn.Module):

    def __init__(self, opts):
        super(StyleCLIPMapper, self).__init__()
        self.opts = opts
        self.mapper = self.set_mapper()


    def set_mapper(self):
        if self.opts.mapper_type == 'ThreeLevelsMapper':
            mapper = latent_mappers.ThreeLevelsMapper(self.opts)
        elif self.opts.mapper_type == 'VAEMapper':
            mapper = latent_mappers.VAEMapper(self.opts)
        else:
            raise Exception('{} is not a valid mapper'.format(self.opts.mapper_type))
        return mapper

    def forward(self, x, zero_var=False):
        if self.opts.KL_lambda > 0:
            codes, mu, log_var = self.mapper(x, return_dstr=True, zero_var=zero_var)
            return codes, mu, log_var, None
        else:
            codes = self.mapper(x)
            return codes, None, None, None

class TwoChannelCLIPMapper(nn.Module):

    def __init__(self, opts):
        super(TwoChannelCLIPMapper, self).__init__()
        self.opts = opts
        self.mapper = self.set_mapper()

    def set_mapper(self):
        if self.opts.mapper_type == 'TwoChannelMapper':
            mapper = latent_mappers.TwoChannelMapper(self.opts)
        elif self.opts.mapper_type == 'cVAEMapper':
            mapper = latent_mappers.cVAEMapper(self.opts)
        elif self.opts.mapper_type == 'cVAEdoubleWMapper':
            mapper = latent_mappers.cVAEdoubleWMapper(self.opts)
        else:
            raise Exception('{} is not a valid mapper'.format(self.opts.mapper_type))
        return mapper

    def forward(self, e, x, zero_var=False):
        if self.opts.KL_lambda > 0:
            codes, mu, log_var, prior_mu = self.mapper(e, x, return_dstr=True, zero_var=zero_var)
            return codes, mu, log_var, prior_mu
        else:
            codes = self.mapper(e, x)
            return codes, None, None, None