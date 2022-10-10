import torch.nn as nn
import torch.nn.init as init
import torch
import numpy as np

# class FactorVAE(nn.Module):
#     """Encoder and Decoder architecture for 2D Shapes data."""
#     def __init__(self, z_dim=10, nc=1):
#         super(FactorVAE, self).__init__()
#         self.z_dim = z_dim
#         self.encoder = nn.Sequential(
#             nn.Conv2d(nc, 32, 4, 2, 1),
#             nn.ReLU(True),
#             nn.Conv2d(32, 32, 4, 2, 1),
#             nn.ReLU(True),
#             nn.Conv2d(32, 64, 4, 2, 1),
#             nn.ReLU(True),
#             nn.Conv2d(64, 64, 4, 2, 1),
#             nn.ReLU(True),
#             nn.Conv2d(64, 128, 4, 1),
#             nn.ReLU(True),
#             nn.Conv2d(128, 2*z_dim, 1)
#         )

#         decode_dim = z_dim
#         self.decoder = nn.Sequential(
#             nn.Conv2d(decode_dim, 128, 1),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(128, 64, 4),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(64, 64, 4, 2, 1),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(64, 32, 4, 2, 1),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(32, 32, 4, 2, 1),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(32, nc, 4, 2, 1),
#         )
#         self.weight_init()
 

#     def weight_init(self, mode='normal'):
#         if mode == 'kaiming':
#             initializer = kaiming_init
#         elif mode == 'normal':
#             initializer = normal_init

 

#         for block in self._modules:
#             for m in self._modules[block]:
#                 initializer(m)

 

#     def reparametrize(self, mu, logvar):
#         std = logvar.mul(0.5).exp_()
#         eps = std.data.new(std.size()).normal_()
#         return eps.mul(std).add_(mu)

 

#     def forward(self, x, no_dec=False):
#         stats = self.encoder(x)
#         mu = stats[:, :self.z_dim]
#         logvar = stats[:, self.z_dim:]
#         z = self.reparametrize(mu, logvar)

 

#         if no_dec:
#             return z.squeeze()
#         elif self.group:
#             real = torch.sin(2*np.pi*z/self.N)
#             imag = torch.cos(2*np.pi*z/self.N)
#             cm_z = torch.cat([real,imag],dim=1)

 

#             x_recon = self.decoder(cm_z).view(x.size())
#             return x_recon, mu, logvar, z.squeeze()
#         else:
#             x_recon = self.decoder(z).view(x.size())
#             return x_recon, mu, logvar, z.squeeze()
#     def _encode(self, x):
#         return self.encoder(x)

 

#     def _decode(self, z):
#         return self.decoder(z)

class FactorVAE(nn.Module):
    """Encoder and Decoder architecture for 2D Shapes data."""
    def __init__(self, z_dim=10, nc=1, N=10, group = True):
        super(FactorVAE, self).__init__()
        self.z_dim = z_dim
        self.N = N
        self.group = group
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

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

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
            real = torch.sin(2*np.pi*z/self.N)
            imag = torch.cos(2*np.pi*z/self.N)
            cm_z = torch.cat([real,imag],dim=1)

            x_recon = self.decoder(cm_z).view(x.size())
            return x_recon, mu, logvar, z.squeeze()
        else:
            x_recon = self.decoder(z).view(x.size())
            return x_recon, mu, logvar, z.squeeze()
    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
            
class LatentShiftPredictor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LatentShiftPredictor, self).__init__()
        self.type_estimator = nn.Linear(in_dim, np.product(out_dim))
        # self.shift_estimator = nn.Linear(512, 1)

    def forward(self, x1, x2):
        features = torch.cat([x1, x2], dim = 1)
        logits = self.type_estimator(features)
        # shift = self.shift_estimator(features)

        # return logits, shift.squeeze()
        return logits