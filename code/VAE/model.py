"""
Implemented by Fan Fei on May 2020.
"""
import torch
from torch import nn
import torch.nn.functional as F
import sys
sys.path.append(r"../")
from VAE.config import config


def weights_init(m):
    # initialize the weight of a layer as DCGAN paper
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, config.init_var)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, config.init_var)
        nn.init.constant_(m.bias, 0.0)
        

class Encoder(nn.Module):
    """
    
    """
    def __init__(self):
        """
        Almost the same architecture as DCGAN discriminator except the output layer.
        """
        super(Encoder, self).__init__()
        nz = config.nz
        ndf = config.ndf
        nc = config.nc
        # size: (nc x 64 x 64)
        self.C1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        # size -> (ndf x 32 x 32)
        self.C2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.BN2 = nn.BatchNorm2d(ndf * 2)
        # size -> (ndf*2 x 16 x 16)
        self.C3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.BN3 = nn.BatchNorm2d(ndf * 4)
        # size -> (ngf*4 x 8 x 8)
        self.C4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.BN4 = nn.BatchNorm2d(ndf * 8)
        # size -> (ngf*8 x 4 x 4)
        self.C5 = nn.Conv2d(ndf * 8, nz * 2, 4, 1, 0, bias=False)
        # size -> (nz * 2) (mu and log_sigma)
        self.apply(weights_init)
        self.scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.scale_shift = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=config.learning_rate,
                                          betas=(config.beta1, 0.999))
    
    def forward(self, x):
        """
        Args:
            x (torch tensor):
        Return:
            mu (torch tensor):
                of shape (nz, )
                the mean vector of gaussian distribution Z|x
            log_sigma (torch tensor):
                of shape (nz, )
                For computational advantage we choose the covariance matrix to be diagonal,
                    and sigma is the collected values on diagonal.
        """
        h = F.leaky_relu(self.C1(x), config.slope)
        h = F.leaky_relu(self.BN2(self.C2(h)), config.slope)
        h = F.leaky_relu(self.BN3(self.C3(h)), config.slope)
        h = F.leaky_relu(self.BN4(self.C4(h)), config.slope)
        mu, log_sigma = torch.split(self.C5(h), config.nz, dim=1)
        mu = torch.tanh(mu) # the pixel value is between [-1, 1]
        log_sigma = self.scale * torch.tanh(log_sigma) + self.scale_shift
        return mu, log_sigma
        
        
        
class Decoder(nn.Module):
    """
    Decoder in Variational AutoEncoder.
    """
    def __init__(self):
        """
        Totally identical architecture as DCGAN generator.
        """
        super(Decoder, self).__init__()
        nz = config.nz
        ngf = config.ngf
        nc = config.nc
        # size: (nz)
        self.CT1 = nn.ConvTranspose2d(in_channels=nz, out_channels=ngf * 8,
                                      kernel_size=4, stride=1,
                                      padding=0, bias=False)
        # No bias because the subsequent normalization will cancel it
        self.BN1 = nn.BatchNorm2d(ngf * 8)
        # size -> (ngf*8 x 4 x 4)
        self.CT2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.BN2 = nn.BatchNorm2d(ngf * 4)
        # size -> (ngf*4 x 8 x 8)
        self.CT3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.BN3 = nn.BatchNorm2d(ngf * 2)
        # size -> (ngf*2 x 16 x 16)
        self.CT4 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
        self.BN4 = nn.BatchNorm2d(ngf)
        # size -> (ngf x 32 x 32)
        self.CT5 = nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)
        # size -> (nc x 64 x 64)
        self.apply(weights_init)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=config.learning_rate,
                                          betas=(config.beta1, 0.999))
    
    def forward(self, z):
        """
        Args:
            z (torch tensor):
                of shape (nz,)
                the latent vector
        Return:
            x (torch tensor):
                of shape (nc, img_size, img_size)
                generated data point
        """
        x = F.leaky_relu(self.BN1(self.CT1(z)), config.slope)
        x = F.leaky_relu(self.BN2(self.CT2(x)), config.slope)
        x = F.leaky_relu(self.BN3(self.CT3(x)), config.slope)
        x = F.leaky_relu(self.BN4(self.CT4(x)), config.slope)
        x = torch.tanh(self.CT5(x))
        return x