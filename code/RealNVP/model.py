import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.resnet import BasicBlock

from RealNVP.config import config


def weights_init(m):
	# initialize the weight of a layer as DCGAN
	if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
		nn.init.normal_(m.weight, 0.0, config.init_var)
	elif isinstance(m, nn.BatchNorm2d):
		nn.init.normal_(m.weight, 1.0, config.init_var)
		nn.init.constant_(m.bias, 0.0)

def batch_stat(x):
	"""
	Compute batch statistics (mean and variance)
	"""
	mean = torch.mean(x, dim=[0, 2, 3], keepdim=True)
	var = torch.mean((x-mean)**2, dim=[0, 2, 3], keepdim=True)
	return mean, var

class Residual_Module(nn.Module):
	def __init__(self, in_dim, dim, out_dim, nblocks = 8):
		"""
		Initializes a residual module.
		Args:
			To be added.
		"""
		super(Residual_Module, self).__init__()
		self.nblocks = nblocks
		assert self.nblocks > 0
		self.in_block = nn.utils.weight_norm(
			nn.Conv2d(in_dim, dim, (3, 3), stride=1, padding=1, bias=True)
		)
		self.core_blocks = nn.ModuleList(
			[BasicBlock(dim, dim) for _ in range(nblocks)]
		)
		self.out_block = nn.utils.weight_norm(
			nn.Conv2d(dim, out_dim, (1, 1), stride=1, padding=0, bias=True),
		)
	
	def forward(self, x):
		x = self.in_block(x)
		for core_block in self.core_blocks:
			x = core_block(x)
		return self.out_block(x)


class Coupling_Layer(nn.Module):
	"""
	Coupling Layer used in RealNVP.
	Property: specific masking (e.g. a instance of checkerboard masking)
	"""
	def __init__(self, mask, in_out_dim=None, mid_dim=None):
		super(Coupling_Layer, self).__init__()
		self.mask = mask
		if in_out_dim is None:
			in_out_dim = config.nc
		if mid_dim is None:
			mid_dim = config.nf
		# size: (nc x 64 x 64)
		self.in_bn = nn.BatchNorm2d(in_out_dim)
		self.core = nn.Sequential(
			nn.ReLU(),
			Residual_Module(in_out_dim, mid_dim, 2*in_out_dim)
		)
		self.core.apply(weights_init)
		self.scale = nn.Parameter(torch.zeros(1), requires_grad=True)
		self.scale_shift = nn.Parameter(torch.zeros(1), requires_grad=True)
		# self.out_bn = nn.BatchNorm2d(in_out_dim)
		# optimizer will be added in the class "Flow"
		
	def forward(self, x, generating = False):
		"""
		Params:
			x: (torch Tensor) input, such as picture or latent vector or interim vector
				shape = (nc x image_size x image_size)
				(to be masked using self.mask)
			generating (bool):
				True if is generating (reversed)
		Returns:
			x (torch Tensor):
				input after affine transform
			log_diag_J (torch Tensor):
				logarithm of diagonal of Jacobian matrix df/dx
				(assume x is a 1-d vector)
		"""
		[B, C, _, _] = list(x.size())
		x_ = x * self.mask
		(log_rescale, shift) = self.core(self.in_bn(x_)).split(C, dim=1)
		# for stability
		log_rescale = self.scale * torch.tanh(log_rescale) + self.scale_shift
		
		log_rescale = log_rescale * (1.0 - self.mask)
		shift = shift * (1.0 - self.mask)
		
		log_diag_J = log_rescale
		# masked logscaling and shift are 0, so applying this step on them won't have any effect
		if generating:
			x = (x - shift) / torch.exp(log_rescale)
		else:
			x = x * torch.exp(log_rescale) + shift
		#do NOT use out bn
		#_, var = batch_stat(x)
		#x = self.out_bn(x) * (1.0 - self.mask) + x * self.mask
		#log_diag_J = log_diag_J - 0.5 * torch.log(var + 1e-5) * (1.0 - self.mask)
		return x, log_diag_J
	
	
class Flow(nn.Module):
	"""
	The inference function f, s.t. z = f(x)
		where x is a picture and z is a latent vector
	The generative function g = f ^ -1
	
	Use architecture described in NICE paper
		rather the multiscale architecture in RealNVP
		(for the purpose of comparing between different kind of models)
	
	The prior distribution Z is set to be the isotropic high-dimensional Gaussian
	"""
	def __init__(self):
		super(Flow, self).__init__()
		self.nCouplingLayers = config.nCouplingLayers
		self.CouplingLayers = []
		# obtaining mask by following steps
		mask = torch.arange(0, config.image_size)
		# use broadcasting to obtain a tensor of shape (image_size x image_size)
		mask = torch.arange(0, config.image_size).reshape(-1, 1) + mask
		# mod 2 to obtain checkerboard masking
		mask = torch.fmod(mask, 2).float().to(device=config.device)
		
		for i in range(self.nCouplingLayers):
			self.CouplingLayers.append(Coupling_Layer(mask, ))
			# alternating pattern
			mask = 1 - mask
		
		self.CouplingLayers = nn.ModuleList(self.CouplingLayers)
		self.optimizer = torch.optim.Adam(self.parameters(),
										  lr=config.learning_rate,
										  weight_decay=config.l2_weight_decay)
		
		self.forward = self.infer # alias
	
		
	def infer(self, x):
		"""
		inference function f
			f(x) = z
		Params:
			x: (torch Tensor)
				a data of picture (forward, about to apply f)
				shape = (nc x image_size x image_size)
		Return:
			z: (torch Tensor)
				latent vector,
				shape = (nc x image_size x image_size)
			log_diag_J: (torch Tensor)
				log of Jacobian diagonal, which is
				sum of sum_i{s_i} in all coupling layers
		"""
		z = x
		log_diag_J = torch.zeros_like(z)
		for i in range(self.nCouplingLayers):
			z, inc = self.CouplingLayers[i](z)
			log_diag_J += inc
		
		return z, log_diag_J
			
		
	
	def generate(self, z):
		"""
		reverse function of inference
		not used for training
		Params:
			z: (torch Tensor)
			 	a latent vector (reverse, about to apply g = f^-1)
				shape = (nc x image_size x image_size)
		Return:
			x: (torch Tensor)
				the picture generated according to latent vector z,
				shape = (nc x image_size x image_size)
		"""
		x = z
		for i in reversed(range(self.nCouplingLayers)):
			x, _ = self.CouplingLayers[i](x, generating=True)
			
		return x