import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.contiguous().view(self.size)
    
class BetaTCVAE(nn.Module):
    """Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018)."""

    def __init__(self, z_dim=10, nc=1, N=10, group = True):
        super(BetaTCVAE, self).__init__()
        self.nc = nc
        self.z_dim = z_dim
        self.N = N
        self.group = group
        self.register_buffer('prior_params', torch.zeros(self.z_dim, 2))

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  8,  8
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  4,  4
            nn.ReLU(True),
            View((-1, 32*4*4)),                  # B, 512
            nn.Linear(32*4*4, 256),              # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )
        if self.group:
            decode_dim = 2*z_dim
        else:
            decode_dim = z_dim
        self.decoder = nn.Sequential(
            nn.Linear(decode_dim, 256),          # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, 32*4*4),              # B, 512
            nn.ReLU(True),
            View((-1, 32, 4, 4)),                # B,  32,  4,  4
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1), # B,  nc, 64, 64
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        
        if self.group:
            real = torch.sin(2*np.pi*z/self.N)
            imag = torch.cos(2*np.pi*z/self.N)
            cm_z = torch.cat([real,imag],dim=1)

            x_recon = self._decode(cm_z).view(x.size())
        else:
            x_recon = self._decode(z).view(x.size())
        return x_recon, mu, logvar, z

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)
    
    def _get_prior_params(self, batch_size=1):
        expanded_size = (batch_size,) + self.prior_params.size()
        prior_params = Variable(self.prior_params.expand(expanded_size))
        return prior_params.cuda()

    def zcomplex(self, z):
        real = torch.sin(2*np.pi*z/self.N)
        imag = torch.cos(2*np.pi*z/self.N)
        return torch.cat([real,imag],dim=1)

class FactorVAE1(nn.Module):
    """Encoder and Decoder architecture for 2D Shapes data."""
    def __init__(self, z_dim=10, nc=1, N=10, group = True):
        super(FactorVAE1, self).__init__()
        self.z_dim = z_dim
        self.N = N
        self.group = group
        self.omega_0 = 30
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 2*z_dim, 1)
        )
        if group:
            decode_dim = 2*z_dim
        else:
            decode_dim = z_dim
        self.decoder = nn.Sequential(
            nn.Conv2d(decode_dim, 128, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),
        )
        self.weight_init()
        if group:
            self.encoder[10].weight.data.uniform_(-np.sqrt(6 / 128), 
                                                np.sqrt(6 / 128))
            self.decoder[0].weight.data.uniform_(-np.sqrt(6 / decode_dim) / self.omega_0, 
                                                np.sqrt(6 / decode_dim) / self.omega_0)
            # pass

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init
        elif mode == 'sine':
            initializer = sine_init
        

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)


    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x, no_dec=False):
        stats = self.encoder(x)
        mu = stats[:, :self.z_dim]
        logvar = stats[:, self.z_dim:]
        z = self.reparametrize(mu, logvar)

        if no_dec:
            return z.squeeze()
        elif self.group:
            cm_z = self.zcomplex(z)

            x_recon = self.decoder(cm_z).view(x.size())
            return x_recon, mu, logvar, z.squeeze()
        else:
            x_recon = self.decoder(z).view(x.size())
            return x_recon, mu, logvar, z.squeeze()

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

    def zcomplex(self, z):
        real = self.omega_0*torch.sin(2*np.pi*z/self.N)
        imag = self.omega_0*torch.cos(2*np.pi*z/self.N)
        return torch.cat([real,imag],dim=1)
    
class GANbaseline(nn.Module):
    """Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018)."""

    def __init__(self, z_dim=10, nc=1, N=10, group = True):
        super(GANbaseline, self).__init__()
        self.nc = nc
        self.z_dim = z_dim
        self.N = N
        self.group = group

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 64, 7, 1, 3),          # B,  32, 32, 32
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),          # B,  32, 16, 16
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),          # B,  32,  8,  8
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 4, 2, 1),          # B,  32,  4,  4
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 4, 2, 1),          # B,  32,  4,  4
            nn.LeakyReLU(),
            View((-1, 4096)),                  # B, 512
            nn.Linear(4096, 256),              # B, 256
            nn.LeakyReLU(),
            nn.Linear(256, 256),                 # B, 256
            nn.LeakyReLU(),
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )
        if self.group:
            decode_dim = 2*z_dim
        else:
            decode_dim = z_dim
        self.decoder = nn.Sequential(
            nn.Linear(decode_dim, 256),          # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, 4096),              # B, 512
            nn.ReLU(True),
            View((-1, 256, 4, 4)),                # B,  32,  4,  4
            nn.ConvTranspose2d(256, 256, 4, 2, 1), # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1), # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(64, nc, 7, 1, 3), # B,  nc, 64, 64
        )
        self.weight_init()
        self.register_buffer('prior_params', torch.zeros(self.z_dim, 2))

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        
        if self.group:
            real = torch.sin(2*np.pi*z/self.N)
            imag = torch.cos(2*np.pi*z/self.N)
            cm_z = torch.cat([real,imag],dim=1)

            x_recon = self._decode(cm_z).view(x.size())
        else:
            x_recon = self._decode(z).view(x.size())
        return x_recon, mu, logvar, z

    def _get_prior_params(self, batch_size=1):
        expanded_size = (batch_size,) + self.prior_params.size()
        prior_params = Variable(self.prior_params.expand(expanded_size))
        return prior_params.cuda()

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)
        
class GANbaseline2(nn.Module):
    """Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018)."""

    def __init__(self, z_dim=10, nc=1, N=10, group = True):
        super(GANbaseline2, self).__init__()
        self.nc = nc
        self.z_dim = z_dim
        self.N = N
        self.group = group

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 64, 7, 1, 3),          # B,  32, 32, 32
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),          # B,  32, 16, 16
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),          # B,  32,  8,  8
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 4, 2, 1),          # B,  32,  4,  4
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 4, 2, 1),          # B,  32,  4,  4
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 4, 2, 1),          # B,  32,  4,  4
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 4, 2, 1),          # B,  32,  4,  4
            nn.LeakyReLU(),
            View((-1, 4096)),                  # B, 512
            nn.Linear(4096, 256),              # B, 256
            nn.LeakyReLU(),
            nn.Linear(256, 256),                 # B, 256
            nn.LeakyReLU(),
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )
        if self.group:
            decode_dim = 2*z_dim
        else:
            decode_dim = z_dim
        self.decoder = nn.Sequential(
            nn.Linear(decode_dim, 256),          # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, 4096),              # B, 512
            nn.ReLU(True),
            View((-1, 256, 4, 4)),                # B,  32,  4,  4
            nn.ConvTranspose2d(256, 256, 4, 2, 1), # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1), # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1), # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1), # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(64, nc, 7, 1, 3), # B,  nc, 64, 64
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        
        if self.group:
            real = torch.sin(2*np.pi*z/self.N)
            imag = torch.cos(2*np.pi*z/self.N)
            cm_z = torch.cat([real,imag],dim=1)

            x_recon = self._decode(cm_z).view(x.size())
        else:
            x_recon = self._decode(z).view(x.size())
        return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)
    
    def zcomplex(self, z):
        real = torch.sin(2*np.pi*z/self.N)
        imag = torch.cos(2*np.pi*z/self.N)
        return torch.cat([real,imag],dim=1)
    
    
class GANbaseline3(nn.Module):
    """Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018)."""

    def __init__(self, z_dim=10, nc=1, N=10, group = True):
        super(GANbaseline3, self).__init__()
        self.nc = nc
        self.z_dim = z_dim
        self.N = N
        self.group = group

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 64, 7, 1, 3),          # B,  32, 32, 32
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),          # B,  32, 16, 16
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),          # B,  32,  8,  8
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 4, 2, 1),          # B,  32,  4,  4
            nn.LeakyReLU(),
            View((-1, 4096)),                  # B, 512
            nn.Linear(4096, 256),              # B, 256
            nn.LeakyReLU(),
            nn.Linear(256, 256),                 # B, 256
            nn.LeakyReLU(),
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )
        if self.group:
            decode_dim = 2*z_dim
        else:
            decode_dim = z_dim
        self.decoder = nn.Sequential(
            nn.Linear(decode_dim, 256),          # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, 4096),              # B, 512
            nn.ReLU(True),
            View((-1, 256, 4, 4)),                # B,  32,  4,  4
            nn.ConvTranspose2d(256, 256, 4, 2, 1), # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(64, nc, 7, 1, 3), # B,  nc, 64, 64
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        
        if self.group:
            real = torch.sin(2*np.pi*z/self.N)
            imag = torch.cos(2*np.pi*z/self.N)
            cm_z = torch.cat([real,imag],dim=1)

            x_recon = self._decode(cm_z).view(x.size())
        else:
            x_recon = self._decode(z).view(x.size())
        return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)
class BetaTCVAE(nn.Module):
    """Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018)."""

    def __init__(self, z_dim=10, nc=1, N=10, group = True):
        super(BetaTCVAE, self).__init__()
        self.nc = nc
        self.z_dim = z_dim
        self.N = N
        self.group = group
        self.register_buffer('prior_params', torch.zeros(self.z_dim, 2))

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  8,  8
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  4,  4
            nn.ReLU(True),
            View((-1, 32*4*4)),                  # B, 512
            nn.Linear(32*4*4, 256),              # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )
        if self.group:
            decode_dim = 2*z_dim
        else:
            decode_dim = z_dim
        self.decoder = nn.Sequential(
            nn.Linear(decode_dim, 256),          # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, 32*4*4),              # B, 512
            nn.ReLU(True),
            View((-1, 32, 4, 4)),                # B,  32,  4,  4
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1), # B,  nc, 64, 64
        )

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        
        if self.group:
            real = torch.sin(2*np.pi*z/self.N)
            imag = torch.cos(2*np.pi*z/self.N)
            cm_z = torch.cat([real,imag],dim=1)

            x_recon = self._decode(cm_z).view(x.size())
        else:
            x_recon = self._decode(z).view(x.size())
        return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)
    
    def _get_prior_params(self, batch_size=1):
        expanded_size = (batch_size,) + self.prior_params.size()
        prior_params = Variable(self.prior_params.expand(expanded_size))
        return prior_params.cuda()
    def zcomplex(self, z):
        real = torch.sin(2*np.pi*z/self.N)
        imag = torch.cos(2*np.pi*z/self.N)
        return torch.cat([real,imag],dim=1)

class FactorVAE1(nn.Module):
    """Encoder and Decoder architecture for 2D Shapes data."""
    def __init__(self, z_dim=10, nc=1, N=10, group = True):
        super(FactorVAE1, self).__init__()
        self.z_dim = z_dim
        self.N = N
        self.group = group
        self.omega_0 = 30
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 2*z_dim, 1)
        )
        if group:
            decode_dim = 2*z_dim
        else:
            decode_dim = z_dim
        self.decoder = nn.Sequential(
            nn.Conv2d(decode_dim, 128, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),
        )
        # self.weight_init()
        if group:
            self.encoder[10].weight.data.uniform_(-np.sqrt(6 / 128), 
                                                np.sqrt(6 / 128))
            self.decoder[0].weight.data.uniform_(-np.sqrt(6 / decode_dim) / self.omega_0, 
                                                np.sqrt(6 / decode_dim) / self.omega_0)
            # pass

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init
        elif mode == 'sine':
            initializer = sine_init
        

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)


    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x, no_dec=False):
        stats = self.encoder(x)
        mu = stats[:, :self.z_dim]
        logvar = stats[:, self.z_dim:]
        z = self.reparametrize(mu, logvar)

        if no_dec:
            return z.squeeze()
        elif self.group:
            cm_z = self.zcomplex(z)

            x_recon = self.decoder(cm_z).view(x.size())
            return x_recon, mu, logvar, z.squeeze()
        else:
            x_recon = self.decoder(z).view(x.size())
            return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

    def zcomplex(self, z):
        real = self.omega_0*torch.sin(2*np.pi*z/self.N)
        imag = self.omega_0*torch.cos(2*np.pi*z/self.N)
        return torch.cat([real,imag],dim=1)

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()

"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
from torchvision.models.vgg import vgg19


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(
        self, block, num_blocks, in_channel=3, zero_init_residual=False, size=64
    ):
        super(ResNet, self).__init__()
        self.in_planes = 64
        second_stride = 2 if size > 32 else 1
        self.conv1 = nn.Conv2d(
            in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=second_stride)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


class CNN_Encoder(nn.Module):
    def __init__(self, in_channel=3, size=64):
        super().__init__()
        init_stride = 2 if size == 64 else 1
        init_padding = 1 if size == 64 else 2
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, 32, 4, init_stride, init_padding),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        return torch.flatten(self.encoder(x), 1)


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def cnn_encoder(**kwargs):
    return CNN_Encoder(**kwargs)


class ImagenetNormalize(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "mean", torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x):
        return (x - self.mean) / self.std


class VGGFeatures(nn.Module):
    """
    change of normalization (-1, 1) -> imagenet
    conv4_3 features
    """

    def __init__(self):
        super().__init__()
        self.normalize = ImagenetNormalize()
        self.vgg_features = vgg19(pretrained=True).features[:24]

    def forward(self, x):
        x = (x + 1) / 2
        x = self.normalize(x)
        return self.vgg_features(x)


class InDomainEncoder(nn.Module):
    def __init__(self, generator, discriminator, n_latent=10):
        super().__init__()
        self.generator = generator
        for p in self.generator.parameters():
            p.requires_grad_(False)

        self.generator.eval()
        self.register_buffer("mean_latent", self.generator.mean_latent(4096))
        self.discriminator = discriminator
        self.encoder = nn.Sequential(
            resnet34(),
            nn.ReLU(),
            nn.Linear(512, n_latent * 512),
            nn.BatchNorm1d(n_latent * 512),
        )
        self.n_latent = n_latent

    def forward(self, x, **generator_kwargs):
        # x - real data

        Ex = self.encoder(x).view(-1, self.n_latent, 512).squeeze(1) + self.mean_latent
        GEx = self.generator([Ex], **generator_kwargs)[0]
        DGEx = self.discriminator(GEx)
        Dx = self.discriminator(x)

        output = {}
        output["Ex"] = Ex
        output["GEx"] = GEx
        output["DGEx"] = DGEx
        output["Dx"] = Dx

        return output


class FactorRegressor(nn.Module):
    def __init__(
        self, discrete_cardinality, backbone="resnet18", f_size=256, **bb_kwargs
    ):
        super().__init__()
        self.backbone = nn.Sequential(
            globals()[backbone](**bb_kwargs),
            nn.ReLU(),
            nn.Linear(512, f_size),
            nn.ReLU(),
        )
        self.classification_heads = nn.ModuleList(
            [nn.Linear(f_size, d) for d in discrete_cardinality]
        )

    def forward(self, x):
        features = self.backbone(x)
        discr = [head(features) for head in self.classification_heads]
        return discr


class LatentRegressor(nn.Module):
    def __init__(self, latent_dimension, backbone="resnet18", f_size=256, **bb_kwargs):
        super().__init__()

        features = 512 if "resnet" in backbone else 1024

        self.backbone = nn.Sequential(
            globals()[backbone](**bb_kwargs),
            nn.ReLU(),
            nn.Linear(features, f_size),
            nn.ReLU(),
        )
        self.latent_head = nn.Linear(f_size, latent_dimension)

    def forward(self, x):
        features = self.backbone(x)
        latent = self.latent_head(features)
        return latent


class EasyRectifier(nn.Module):
    def __init__(self, generator, regressor, device, **generator_kwargs):
        super().__init__()

        self.generator = generator
        self.regressor = regressor
        self.generator_kwargs = generator_kwargs
        self.device = device
        self.generator.eval()
        self.regressor.eval()

        for p in self.generator.parameters():
            p.requires_grad_(False)

        for p in self.regressor.parameters():
            p.requires_grad_(False)

        self.dir = nn.Parameter(torch.randn(1, self.generator.style_dim))

    def forward(self, batch_size=64, proj=None):

        # proj K x 512

        with torch.no_grad():
            if proj is not None:
                self.dir.data = self.dir.data - self.dir.data @ proj

            self.dir.data = nn.functional.normalize(self.dir.data, p=2, dim=1)

        z = torch.randn(batch_size, self.generator.style_dim).to(self.device)
        alpha = 2 * (2 * torch.rand(batch_size, 1).to(self.device) - 1)

        with torch.no_grad():
            w = self.generator.style(z)

            predicts_orig = self.regressor(
                torch.clamp(self.generator([w], **self.generator_kwargs)[0], -1, 1)
            )

        predicts_shifted = self.regressor(
            torch.clamp(
                self.generator([w + alpha * self.dir], **self.generator_kwargs)[0],
                -1,
                1,
            )
        )

        return predicts_orig, predicts_shifted


class FullCrossEntropy(nn.Module):
    def forward(self, x, y):
        return F.kl_div(F.log_softmax(x, dim=1), y, reduction="batchmean")