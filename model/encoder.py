import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, embd_dim, max_length=32):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.embd_dim = embd_dim
        self.max_length = max_length
        
        # length module.
        self.length_net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(), nn.LayerNorm(1024),
            nn.Linear(1024, 512),
            nn.ReLU(), nn.LayerNorm(512),
            nn.Linear(512, max_length)
        )
        
        # content module with continuous regularizer.
        self.reg_net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(), nn.LayerNorm(1024),
            nn.Linear(1024, 512),
            nn.ReLU(), nn.LayerNorm(512),
            nn.Linear(512, int(2*embd_dim))
        )
        self.content_net = nn.Sequential(
            nn.Linear(int(2*embd_dim), 512),
            nn.ReLU(), nn.LayerNorm(512),
            nn.Linear(512, 512),
            nn.ReLU(), nn.LayerNorm(512),
            nn.Linear(512, max_length)
        )

        # initialize weights.
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=0.1)
            nn.init.zeros_(m.bias)
    
    def encode(self, x):
        # length part.
        length_logits = self.length_net(x)
        p_l = torch.distributions.Categorical(logits=length_logits)
        l = p_l.sample()
        
        # continuous latent code part.
        zc_logits = self.reg_net(x)
        mu = zc_logits[:, :self.embd_dim]
        std = F.softplus(zc_logits[:, self.embd_dim:] - 5, beta=1) + 1e-6

        p_zc = torch.distributions.Normal(mu, std)
        zc = p_zc.rsample()

        # content part.
        content_logits = self.content_net(zc_logits)
        p_zt = torch.distributions.Bernoulli(logits=content_logits)
        zt = p_zt.sample()

        return l, p_l, zt, p_zt, zc, p_zc
    
    def param(self):
        return list(self.parameters())
