#Code borrowd from https://github.com/orpatashnik/StyleCLIP/blob/main/criteria/clip_loss.py
import torch
import clip


class CLIPLoss(torch.nn.Module):

    def __init__(self, opts):
        super(CLIPLoss, self).__init__()
        self.opts = opts
        self.model, self.preprocess = clip.load("ViT-B/16", device=self.opts.device)
        self.model.eval()
        self.upsample = torch.nn.Upsample(scale_factor=7)
        if self.opts.restore_resize:
            self.kk = int(self.opts.output_size/32)
            self.avg_pool = torch.nn.AvgPool2d(kernel_size=self.opts.output_size // self.kk)
        else:
            self.kk = int(256/8)
            self.avg_pool = torch.nn.AvgPool2d(kernel_size=256 // self.kk)

    def forward(self, recon, orig_features):
        recon = self.avg_pool(self.upsample(recon))
        orig_features = orig_features/(orig_features.norm(dim=1,keepdim=True)+1e-8)
        recon_features = self.model.encode_image(recon)
        recon_features = recon_features/(recon_features.norm(dim=1,keepdim=True)+1e-8)
        similarity = (orig_features*recon_features).sum(dim=1)
        return (1-similarity).mean()