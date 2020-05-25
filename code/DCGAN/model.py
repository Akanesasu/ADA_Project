import torch
from torch import nn
import torch.nn.functional as F
import sys
sys.path.append(r"../")
from DCGAN.config import config


def weights_init(m):
	# initialize the weight of a layer as DCGAN paper
	if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
		nn.init.normal_(m.weight, 0.0, config.init_var)
	elif isinstance(m, nn.BatchNorm2d):
		nn.init.normal_(m.weight, 1.0, config.init_var)
		nn.init.constant_(m.bias, 0.0)


class Generator(nn.Module):
	"""
	Generator in DCGAN
	"""

	def __init__(self, nz, ngf, nc):
		super(Generator, self).__init__()
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

	def forward(self, input):
		h = F.relu(self.BN1(self.CT1(input)))
		h = F.relu(self.BN2(self.CT2(h)))
		h = F.relu(self.BN3(self.CT3(h)))
		h = F.relu(self.BN4(self.CT4(h)))
		h = torch.tanh(self.CT5(h))
		return h


class Discriminator(nn.Module):
	"""
	Discriminator in DCGAN
	"""
	
	def __init__(self, nc, ndf):
		super(Discriminator, self).__init__()
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
		self.C5 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
		# size -> (1)
		self.apply(weights_init)
		self.optimizer = torch.optim.Adam(self.parameters(),
										  lr=config.learning_rate,
										  betas=(config.beta1, 0.999))
	
	def forward(self, input):
		h = F.leaky_relu(self.C1(input), config.slope)
		h = F.leaky_relu(self.BN2(self.C2(h)), config.slope)
		h = F.leaky_relu(self.BN3(self.C3(h)), config.slope)
		h = F.leaky_relu(self.BN4(self.C4(h)), config.slope)
		h = torch.sigmoid(self.C5(h))
		return h
	