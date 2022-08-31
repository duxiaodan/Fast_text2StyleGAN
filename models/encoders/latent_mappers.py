import torch
from torch import nn
from torch.nn import Module
import copy
import pdb
from models.stylegan2.model import EqualLinear, PixelNorm
torch.autograd.set_detect_anomaly(False)
STYLESPACE_DIMENSIONS = [512 for _ in range(15)] + [256, 256, 256] + [128, 128, 128] + [64, 64, 64] + [32, 32]


class Mapper(Module):

    def __init__(self, opts, latent_dim=512, num_layers=4, input_dim=None):
        super(Mapper, self).__init__()
        # print(opts.lr_mul)
        self.latent_dim = latent_dim
        self.opts = opts
        layers = [PixelNorm()]
        if input_dim is not None:
            pass
        else:
            input_dim = latent_dim
        layers.append(
            EqualLinear(
                input_dim, latent_dim, lr_mul=opts.lr_mul, activation='fused_lrelu'
            )
        )
        for i in range(num_layers-1):
            layers.append(
                EqualLinear(
                    latent_dim, latent_dim, lr_mul=opts.lr_mul, activation='fused_lrelu'
                )
            )
        self.input_dim = input_dim
        self.mapping = nn.Sequential(*layers)


    def forward(self, x):
        x = self.mapping(x)
        return x


class ThreeLevelsMapper(Module):

    def __init__(self, opts):
        super(ThreeLevelsMapper, self).__init__()

        self.opts = opts
        self.mapper_list=nn.ModuleList()
        if not opts.no_coarse_mapper:
            self.mapper_list.append(Mapper(opts))
        if not opts.no_medium_mapper:
            self.mapper_list.append(Mapper(opts))
        if not opts.no_fine_mapper:
            self.mapper_list.append(Mapper(opts))

    def forward(self, x):

        out_list = []
        if not self.opts.no_coarse_mapper:
            for i in range(0,4):
                out_list.append(self.mapper_list[0](x))
        else:
            for i in range(0,4):
                out_list.append(torch.zeros_like(x))
        if not self.opts.no_medium_mapper:
            for i in range(4,8):
                out_list.append(self.mapper_list[1](x))
        else:
            for i in range(4,8):
                out_list.append(torch.zeros_like(x))
        if not self.opts.no_fine_mapper:
            for i in range(8,self.opts.n_styles):
                out_list.append(self.mapper_list[2](x))
        else:
            for i in range(8,self.opts.n_styles):
                out_list.append(torch.zeros_like(x))


        out = torch.stack(out_list, dim=1)

        return out

class VAEMapper(Module):
    def __init__(self, opts, output_dim=512):
        super(VAEMapper, self).__init__()
        self.opts = opts
        self.output_dim = output_dim
        self.mapper = Mapper(opts)
        self.latent_dim = self.mapper.latent_dim
        self.to_mu = nn.Linear(self.latent_dim, self.output_dim)
        self.to_var = nn.Linear(self.latent_dim, self.output_dim)
        
    def encode(self, x):
        x = self.mapper(x)
        return self.to_mu(x), self.to_var(x)
    
    def reparameterize(self, mu, log_var, zero_var=False):
        std = torch.exp(log_var*0.5)
        eps = torch.randn_like(std)
        if zero_var == True:
            return mu
        else:
            return mu + eps * std
    
    def forward(self, x, return_dstr=False, zero_var=False):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var, zero_var=zero_var)
        if return_dstr:
            return z, mu, log_var
        else:
            return z


class TwoChannelMapper(Module):
    def __init__(self, opts, output_dim=512, input_hw=256):
        super(TwoChannelMapper, self).__init__()
        self.opts = opts
        self.output_dim = output_dim
        self.bottleNeck_dim = self.opts.bottleNeck_dim
        self.input_hw = input_hw
        self.CLIPMapper = Mapper(opts)
        self.latent_dim = self.CLIPMapper.latent_dim

        modules = []
        in_channels = 3
        hidden_dims = [32, 64, 128, 256, 512] 
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        self.imageEncoder = nn.Sequential(*modules)
        self.imageEncoder_out_dim = hidden_dims[-1] * int(input_hw/(2**len(hidden_dims)))**2
        self.to_mu = nn.Linear(self.imageEncoder_out_dim, self.bottleNeck_dim)
        self.to_var = nn.Linear(self.imageEncoder_out_dim, self.bottleNeck_dim)

        self.ImageMapper = Mapper(opts,input_dim=self.bottleNeck_dim)

    def encodeImage(self, x):
        x = self.imageEncoder(x)
        x = torch.flatten(x, start_dim=1)
        return self.to_mu(x), self.to_var(x)
    
    def reparameterize(self, mu, log_var, zero_var=False):
        std = torch.exp(log_var*0.5)
        eps = torch.randn_like(std)
        if zero_var == True:
            return mu
        else:
            return mu + eps * std
    
    def forward(self, e, x, return_dstr=False, zero_var=False):
        if x != 'random':
            mu, log_var = self.encodeImage(x)
        else:
            mu = torch.zeros(e.shape[0],self.bottleNeck_dim, device=e.device) 
            log_var = torch.zeros(e.shape[0],self.bottleNeck_dim, device=e.device)
        z = self.reparameterize(mu, log_var, zero_var=zero_var)
        w = self.ImageMapper(z) + self.CLIPMapper(e)
        if return_dstr:
            return w, mu, log_var, None
        else:
            return w

class cVAEMapper(Module):
    def __init__(self, opts, output_dim=512, input_hw=256):
        super(cVAEMapper, self).__init__()
        self.opts = opts
        self.output_dim = output_dim
        self.bottleNeck_dim = self.opts.bottleNeck_dim
        self.input_hw = input_hw
        self.CLIPMapper = Mapper(opts)
        self.latent_dim = self.CLIPMapper.latent_dim

        modules = []
        in_channels = 3
        hidden_dims = [32, 64, 128, 256, 512] 
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        self.imageEncoder_out_dim = hidden_dims[-1] * int(input_hw/(2**len(hidden_dims)))**2
        self.imageEncoder = nn.Sequential(*modules)
        self.fc = nn.Linear(self.imageEncoder_out_dim, 512)
        
        self.MLP1 = Mapper(opts,input_dim=1024)
        self.to_mu = nn.Linear(512, self.bottleNeck_dim)
        self.to_var = nn.Linear(512, self.bottleNeck_dim)
        if self.opts.learn_in_w:
            self.MLP2 = Mapper(opts,input_dim=512+self.bottleNeck_dim)
        else:
            self.MLP2 = nn.ModuleList()
            if not opts.no_coarse_mapper:
                self.MLP2.append(Mapper(opts,input_dim=512+self.bottleNeck_dim))
            if not opts.no_medium_mapper:
                self.MLP2.append(Mapper(opts,input_dim=512+self.bottleNeck_dim))
            if not opts.no_fine_mapper:
                self.MLP2.append(Mapper(opts,input_dim=512+self.bottleNeck_dim))

    def encodeImage(self, e, x):
        x = self.imageEncoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = torch.cat([x,e],dim=-1)
        x = self.MLP1(x)
        return self.to_mu(x), self.to_var(x)
    
    def reparameterize(self, mu, log_var, zero_var=False):
        std = torch.exp(log_var*0.5)
        eps = torch.randn_like(std)

        if zero_var == True:
            return mu
        else:
            return mu + eps * std
    
    def forward(self, e, x, return_dstr=False, zero_var=False):
        prior_mu = None
        if x == 'random':
            mu = torch.zeros(e.shape[0],self.bottleNeck_dim, device=e.device)
            log_var = torch.zeros(e.shape[0],self.bottleNeck_dim, device=e.device)
        else:
            mu, log_var = self.encodeImage(e,x)
        z = self.reparameterize(mu, log_var, zero_var=zero_var)
        z = torch.cat([z,e],dim=-1)
        
        if self.opts.learn_in_w:
            w = self.MLP2(z)
        else:
            w = []
            if not self.opts.no_coarse_mapper:
                for i in range(0,4):
                    w.append(self.MLP2[0](z))
            else:
                for i in range(0,4):
                    w.append(torch.zeros_like(z))
            if not self.opts.no_medium_mapper:
                for i in range(4,8):
                    w.append(self.MLP2[1](z))
            else:
                for i in range(4,8):
                    w.append(torch.zeros_like(z))
            if not self.opts.no_fine_mapper:
                for i in range(8,self.opts.n_styles):
                    w.append(self.MLP2[2](z))
            else:
                for i in range(8,self.opts.n_styles):
                    w.append(torch.zeros_like(z))
            w = torch.stack(w, dim=1)
        
        if return_dstr:
            return w, mu, log_var, prior_mu
        else:
            return w

class cVAEdoubleWMapper(Module):
    def __init__(self, opts, output_dim=512, input_hw=256):
        super(cVAEdoubleWMapper, self).__init__()
        self.opts = opts
        self.output_dim = output_dim
        self.bottleNeck_dim = self.opts.bottleNeck_dim
        assert self.bottleNeck_dim == 512
        self.input_hw = input_hw
        self.CLIPMapper = Mapper(opts)
        self.latent_dim = self.CLIPMapper.latent_dim
        if self.opts.conditional_mu:
            self.priorMapper = Mapper(opts,latent_dim = self.bottleNeck_dim, input_dim = 512)

        modules = []
        in_channels = 3
        hidden_dims = [32, 64, 128, 256, 512] #feature dim = [512, 256, 128, 64, 32]
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        self.imageEncoder_out_dim = hidden_dims[-1] * int(input_hw/(2**len(hidden_dims)))**2
        self.imageEncoder = nn.Sequential(*modules)
        self.fc = nn.Linear(self.imageEncoder_out_dim, 512)
        
        self.MLP1 = Mapper(opts,input_dim=1024)
        self.to_mu = nn.Linear(512, self.bottleNeck_dim)
        self.to_var = nn.Linear(512, self.bottleNeck_dim)
        self.MLP2 = Mapper(opts)
        self.MLP3 = Mapper(opts, input_dim=1024)
        self.create_f()

    def create_f(self):
        self.SGf = Mapper(self.opts,num_layers=8)
        f_ckpt = torch.load(self.opts.stylegan_weights)['g_ema']
        d_filt = {'mapping.'+k[len('style') + 1:]: v for k, v in f_ckpt.items() if k[:len('style')] == 'style'}
        self.SGf.load_state_dict(d_filt)
        for p in self.SGf.parameters():
            p.requires_grad = False    

    def encodeImage(self, e, x):
        x = self.imageEncoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = torch.cat([x,e],dim=-1)
        x = self.MLP1(x)
        return self.to_mu(x), self.to_var(x)
    
    def reparameterize(self, mu, log_var, zero_var=False):
        std = torch.exp(log_var*0.5)
        eps = torch.randn_like(std)
        if zero_var == True:
            return mu
        else:
            return mu + eps * std
    
    def forward(self, e, x, return_dstr=False, zero_var=False):
        prior_mu = None
        if x == 'random':
            mu = torch.zeros(e.shape[0],self.bottleNeck_dim, device=e.device)
            log_var = torch.zeros(e.shape[0],self.bottleNeck_dim, device=e.device)
        elif type(x) is not torch.Tensor:
            assert 'random' == x[:6]
            std_scale = float(x[6:])
            mu = torch.zeros(e.shape[0],self.bottleNeck_dim, device=e.device) 
            log_var = np.log(std_scale**2)+torch.zeros(e.shape[0],self.bottleNeck_dim, device=e.device)
        else:
            mu, log_var = self.encodeImage(e,x)
        z = self.reparameterize(mu, log_var, zero_var=zero_var)
        
        w1 = self.SGf(z)
        w2 = self.MLP2(e)
        w = self.MLP3(torch.cat([w1,w2],dim=-1))
        
        if return_dstr:
            return w, mu, log_var, prior_mu
        else:
            return w