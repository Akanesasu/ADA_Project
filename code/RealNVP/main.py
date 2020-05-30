"""
Changed from
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""
import os
import random
import torch
import time
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

from RealNVP.model import Flow
from RealNVP.config import config


from torch.autograd import Variable


def Show_Results(log_Prior_Prob_list, log_det_list, img_list):
	
	plt.figure(figsize=(10, 5))
	plt.title("Two Parts of p(x) During Training")
	plt.plot(log_Prior_Prob_list, label="Log of p(f(x))")
	plt.plot(log_det_list, label="Log of det(df/dx)")
	log_ll_list = list(np.sum([log_Prior_Prob_list, log_det_list], axis=0))
	plt.plot(log_ll_list, label="Log of p(x)")
	plt.xlabel("iterations")
	plt.ylabel("Objective")
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

def show_reconstructed_images(imgs, rec_imgs):
	# Plot the real images
	plt.figure(figsize=(15, 15))
	plt.subplot(1, 2, 1)
	plt.axis("off")
	plt.title("Real Images")
	plt.imshow(
		np.transpose(vutils.make_grid(imgs[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))
	
	# Plot the fake images from the last epoch
	plt.subplot(1, 2, 2)
	plt.axis("off")
	plt.title("Rec Images")
	plt.imshow(
		np.transpose(vutils.make_grid(rec_imgs[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))
	plt.savefig(config.output_path + "real_and_fake.png")
	plt.show()

def Show_Model(model):
	dummy_input = torch.autograd.Variable(torch.rand(1, 3, 64, 64))
	out = model(dummy_input)
	#make_dot(out, params=dict(model.named_parameters()))

def Train():
	"""
	Training netF (RealNVP).
	Then Show the result by calling Show_Results function.
	"""
	# Lists to keep track of progress
	img_list = []
	log_Prior_Prob_list = []
	log_det_list = []
	fixed_latent_vectors = torch.randn(64, config.nc, config.image_size,
									   config.image_size, device=device)
	iters = 0
	Prior = torch.distributions.normal.Normal(
		loc=torch.zeros(config.nc, config.image_size, config.image_size).to(device=device),
		scale=1)
	# offset = torch.sum(Prior.log_prob(torch.zeros(config.nc, config.image_size, config.image_size)))
	
	print("Starting Training Loop...")
	# For each epoch
	for epoch in range(config.num_epochs):
		# For each batch in the dataloader
		for i, data in enumerate(dataloader, 0):
			##################################
			# Update Flow-Model
			# Discriminator Loss: (maximize)
			# log(p_X(x)) = log(p_Z(f(x))) + log(|det(df/dx)|)
			##################################
			netF.zero_grad()
			# Calculate log(D(x))
			images = data[0].to(device)
			latent_vectors, log_diag_J = netF(images)
			# reconstructed_images = netF.generate(latent_vectors).detach()
			# show_reconstructed_images(images, reconstructed_images)
			# calculate log_prob for every batch
			log_det = torch.mean(torch.sum(log_diag_J, dim=[1, 2, 3]))
			log_prob = torch.mean(torch.sum(Prior.log_prob(latent_vectors),
			                                dim=[1, 2, 3]))
			loss = - log_prob - log_det
			loss.backward()
			# apply gradients
			netF.optimizer.step()
			
			# Output training stats
			if i % 50 == 0:
				print('[%s] [%d/%d][%d/%d]\tlog(P_Z(f(x))): %.4f\tlog(|det(df/dx)|): %.4f\t'
					  'log_likelihood: %.4f'
                      % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
					  epoch, config.num_epochs, i, len(dataloader),
						 log_prob.item(), log_det.item(), -loss.item()))
			
			# Save losses for plotting later
			log_Prior_Prob_list.append(log_prob.item())
			log_det_list.append(log_det.item())
			
			# Check how the generator is doing by saving G's
			# output on fixed latent vector
			if (iters % 500 == 0) or \
					((epoch == config.num_epochs - 1) and (i == len(dataloader) - 1)):
				with torch.no_grad():
					fake_images = netF.generate(fixed_latent_vectors).detach().cpu()
					img_list.append(
						vutils.make_grid(
							fake_images,
							padding=2,
							normalize=True))
			
			iters += 1
	
	return log_Prior_Prob_list, log_det_list, img_list


def Save_Models():
	"""
	Save model weights
	"""
	if not os.path.exists(config.model_output):
		os.makedirs(config.model_output)
	torch.save(netF.state_dict(), config.model_output + "Flow")


if __name__ == '__main__':
	if not os.path.exists(config.output_path):
		os.makedirs(config.output_path)
	
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
	device = torch.device("cuda:3" if (torch.cuda.is_available() and config.ngpu > 0)
						  else "cpu")
	
	netF = Flow().to(device=device)
	
	# Handle multi-gpu if desired
	# if (device.type == 'cuda') and (config.ngpu > 1):
	#	netF = nn.DataParallel(netF, list(range(config.ngpu)))
	
	#Show_Model(netF)
	log_Prior_Prob_list, log_det_list, img_list = Train()
	Show_Results(log_Prior_Prob_list, log_det_list, img_list)
	Save_Models()
