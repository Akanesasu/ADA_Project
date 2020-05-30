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

from VAE.config import config
from VAE.model import Decoder, Encoder


def Show_Results(Recon_Losses, KLdiv_Losses, Total_losses, img_list):
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)
    
    plt.figure(figsize=(10, 5))
    plt.title("Loss During Training")
    plt.plot(Recon_Losses, label="Reconstruction Loss")
    plt.plot(KLdiv_Losses, label="KL Divergence Loss")
    plt.plot(Total_losses, label="Total Loss")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(config.output_path + "losses.png")
    # plt.show()
    
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(img_list[i], (1, 2, 0)), animated=True)]
           for i in range(len(img_list))]
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
    Training decoder and encoder simultaneously.
    Then Show the result by calling Show_Results function.
    """
    # Lists to keep track of progress
    img_list = []
    KLdiv_Losses = []
    Recon_Losses = []
    Total_losses = []
    fixed_latent_vectors = torch.randn(64, config.nz, 1, 1, device=device)
    recov = config.KL_weight_recover
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(config.num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            encoder.zero_grad()
            images = data[0].to(device)
            batch_size = images.size(0)
            # Compute means and variances
            mu, log_sigma = encoder(images)
            sigma = torch.exp(log_sigma)
            # Compute KL Divergences
            # D[N(mu, Diag(sigma) || N(0, 1)]
            # = 1/2 * [ tr(Diag(sigma)) + mu' * mu - nz - log det(Diag(sigma)) ]
            # note that the sigma - log_sigma term in loss will
            # encourage sigma to be 1 (where x - log(x) reaches minimum)
            # we omit the 1/2 constant
            KLDiv_loss = torch.mean(torch.sum(sigma, dim=1)
                                + torch.sum(mu * mu, dim=1)
                                - config.nz - torch.sum(log_sigma, dim=1))
            # sample random latent vectors
            epsilons = torch.normal(0, 1, size=[batch_size, config.nz, 1, 1]).to(device=device)
            # reparametrization trick
            # z = mu + sigma^(1/2) * epsilon
            latent_vectors = epsilons * torch.sqrt(sigma) + mu

            # Get generated images
            decoder.zero_grad()
            # compute log P(x|z)
            # where P is a gaussian whose mean is reconstructed_images
            #     and covariance matrix is sigma (scalar hyperparameter) * I
            # so we can compute the probability density using
            # f(x) = exp( -1/2 * (x-mu)'(x-mu)/sigma )
            # we can view -log_ll as reconstruction loss
            reconstructed_images = decoder(latent_vectors)
            recon_loss = torch.mean(torch.sum((images - reconstructed_images) * \
                                              (images - reconstructed_images), dim=1) / config.sigma)
            
            # sigmoid schedule
            KL_weight = 1 if iters >= recov \
                else 1.0 / (1 + np.exp(-(iters - recov) / (recov / 50)))
            # we want to maximize object_func
            loss = recon_loss + KL_weight * KLDiv_loss
            loss.backward()
            
            # apply gradients
            decoder.optimizer.step()
            encoder.optimizer.step()

            # Output training stats
            if i % 50 == 0:
                print('[%s] [%d/%d][%d/%d]\tTotal Loss: %.4f\tRecons. Loss: %.4f\tKLDiv. Loss: %.4f\t'
                      'KL Weight: %.4f\t'
                      % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      epoch, config.num_epochs, i, len(dataloader),
                         loss.item(), recon_loss.item(), KLDiv_loss.item(), KL_weight))

            # Save losses for plotting later
            Recon_Losses.append(recon_loss.item())
            KLdiv_Losses.append(KLDiv_loss.item())
            Total_losses.append(loss.item())

            # Check how the generator is doing by saving G's
            # output on fixed latent vector
            if (iters % 500 == 0) or \
                    ((epoch == config.num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake_images = decoder(fixed_latent_vectors).detach().cpu()
                    img_list.append(
                        vutils.make_grid(
                            fake_images,
                            padding=2,
                            normalize=True))

            iters += 1

    return Recon_Losses, KLdiv_Losses, Total_losses, img_list

def Save_Models():
    """
    Save model weights
    """
    if not os.path.exists(config.model_output):
        os.makedirs(config.model_output)
    torch.save(decoder.state_dict(), config.model_output + "decoder")
    torch.save(encoder.state_dict(), config.model_output + "encoder")

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
    device = torch.device("cuda:1" if (torch.cuda.is_available() and config.ngpu > 0)
                          else "cpu")
    print("device: ", device)

    decoder = Decoder().to(device=device)
    encoder = Encoder().to(device=device)

    # Handle multi-gpu if desired
    # if (device.type == 'cuda') and (config.ngpu > 1):
    #    netG = nn.DataParallel(netG, list(range(config.ngpu)))

    Recon_Losses, KLdiv_Losses, Total_losses, img_list = Train()
    Show_Results(Recon_Losses, KLdiv_Losses, Total_losses, img_list)
    Save_Models()
