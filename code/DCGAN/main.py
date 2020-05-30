"""
Reference:
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils
import torchvision.utils as vutils
import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import sys
sys.path.append(r"../")

from DCGAN.model import Generator, Discriminator
from DCGAN.config import config


def Show_Results(G_losses, D_losses, img_list):
	if not os.path.exists(config.output_path):
		os.makedirs(config.output_path)
		
	plt.figure(figsize=(10, 5))
	plt.title("Generator and Discriminator Loss During Training")
	plt.plot(G_losses, label="G")
	plt.plot(D_losses, label="D")
	plt.xlabel("iterations")
	plt.ylabel("Loss")
	plt.legend()
	plt.savefig(config.output_path + "losses.png")
	# plt.show()

	fig = plt.figure(figsize=(8, 8))
	plt.axis("off")
	ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)]
		   for i in img_list]
	ani = animation.ArtistAnimation(
		fig, ims, interval=1000, repeat_delay=1000, blit=True)
	ani.save(config.output_path + "process.gif", writer="pillow")
	# HTML(ani.to_jshtml())

	# Grab a batch of real images from the dataloader
	real_batch = next(iter(dataloader))

	# Plot the real images
	plt.figure(figsize=(15, 15))
	plt.subplot(1, 2, 1)
	plt.axis("off")
	plt.title("Real Images")
	plt.imshow(
		np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

	# Plot the fake images from the last epoch
	plt.subplot(1, 2, 2)
	plt.axis("off")
	plt.title("Fake Images")
	plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
	plt.savefig(config.output_path + "real_and_fake.png")
	# plt.show()


def Train():
	"""
	Training netG and netD alternatively.
	Then Show the result by calling Show_Results function.
	"""
	# Lists to keep track of progress
	img_list = []
	G_losses = []
	D_losses = []
	fixed_latent_vectors = torch.randn(64, config.nz, 1, 1, device=device)
	iters = 0
	# alias
	real_label = 1
	fake_label = 0

	print("Starting Training Loop...")
	# For each epoch
	for epoch in range(config.num_epochs):
		# For each batch in the dataloader
		for i, data in enumerate(dataloader, 0):
			##################################
			# Update Discriminator
			# Discriminator Loss: (maximize)
			# log(D(x)) + log(1 - D(G(z)))
			##################################
			netD.zero_grad()
			# Calculate log(D(x))
			real_images = data[0].to(device)  # x_train
			batch_size = real_images.size(0)
			label_real = torch.full((batch_size, ), real_label,
									device=device, dtype=torch.float32)
			pred_real = netD(real_images).view(-1)  # to 1-d tensor
			loss_D_real = F.binary_cross_entropy(pred_real, label_real)
			D_x = pred_real.mean().item()  # for logging stuff
			# Note that BCE Loss is -(y * log(x) + (1-y) * log(1-x))
			# so minimize it is to maximize log(x) when y = 1
			loss_D_real.backward()
			# Calculate log(1 - D(G(z)))
			latent_vectors = torch.randn(
				batch_size, config.nz, 1, 1, device=device)  # 1, 1 for Conv2d
			fake_images = netG(latent_vectors)
			label_fake = torch.full((batch_size, ), fake_label,
									device=device, dtype=torch.float32)
			pred_fake = netD(fake_images).view(-1)
			D_G_z1 = pred_fake.mean().item()
			loss_D_fake = F.binary_cross_entropy(pred_fake, label_fake)
			loss_D_fake.backward()
			# apply gradients
			loss_D = loss_D_real + loss_D_fake
			netD.optimizer.step()

			##################################
			# Update Generator
			# Generator Loss: (maximize)
			# 		log(D(G(z)))
			# not minimizing log(1-D(G(z))) because of
			# providing no sufficient gradients.
			##################################
			netG.zero_grad()
			latent_vectors = torch.randn(
				batch_size, config.nz, 1, 1, device=device)
			fake_images = netG(latent_vectors)
			# we use real label because we want to maximize log(D(G(z)))
			label_real = torch.full((batch_size, ), real_label,
									device=device, dtype=torch.float32)
			pred_fake = netD(fake_images).view(-1)
			D_G_z2 = pred_fake.mean().item()
			loss_G = F.binary_cross_entropy(pred_fake, label_real)
			loss_G.backward()
			# apply gradients
			netG.optimizer.step()

			# Output training stats
			if i % 50 == 0:
				print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t'
					  'D(x): %.4f\tD(G(z)): %.4f / %.4f'
					  % (epoch, config.num_epochs, i, len(dataloader),
						 loss_D.item(), loss_G.item(), D_x, D_G_z1, D_G_z2))

			# Save losses for plotting later
			G_losses.append(loss_G.item())
			D_losses.append(loss_D.item())

			# Check how the generator is doing by saving G's
			# output on fixed latent vector
			if (iters % 500 == 0) or \
					((epoch == config.num_epochs - 1) and (i == len(dataloader) - 1)):
				with torch.no_grad():
					fake_images = netG(fixed_latent_vectors).detach().cpu()
					img_list.append(
						vutils.make_grid(
							fake_images,
							padding=2,
							normalize=True))

			iters += 1
	
	return G_losses, D_losses, img_list

def Save_Models():
	"""
	Save model weights
	"""
	if not os.path.exists(config.model_output):
		os.makedirs(config.model_output)
	torch.save(netG.state_dict(), config.model_output + "Generator")
	torch.save(netD.state_dict(), config.model_output + "Discriminator")

if __name__ == '__main__':
	random.seed(config.manualSeed)
	torch.manual_seed(config.manualSeed)
	# Create the dataset
	dataset = datasets.ImageFolder(root=config.dataroot,
								   transform=transforms.Compose([
									   transforms.Resize(config.image_size),
									   transforms.CenterCrop(
										   config.image_size),
									   transforms.ToTensor(),
									   transforms.Normalize(
										   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
								   ]))
	# Create the dataloader
	dataloader = DataLoader(dataset, batch_size=config.batch_size,
							shuffle=True, num_workers=config.workers)

	# Decide which device to run on
	device = torch.device("cuda:0" if (torch.cuda.is_available() and config.ngpu > 0)
						  else "cpu")


	netG = Generator(config.nz, config.ngf, config.nc).to(device=device)
	netD = Discriminator(config.nc, config.ndf).to(device=device)

	# Handle multi-gpu if desired
	# if (device.type == 'cuda') and (config.ngpu > 1):
	#	netG = nn.DataParallel(netG, list(range(config.ngpu)))

	G_losses, D_losses, img_list = Train()
	Show_Results(G_losses, D_losses, img_list)
	Save_Models()
