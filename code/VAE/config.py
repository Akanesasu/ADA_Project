
class config(object):
	
	manualSeed = 999  		# random seed for reproducibility
	version = 1.0  			# version
	
	workers = 2  			# number of worker threads for loading
	batch_size = 128  		# as DCGAN paper
	image_size = 64  		# images are nc x image_size x image_size
	nc = 3  				# number of channels (3 for RGB)
	nz = 100  				# number of dimensions of latent vector z
	ngf = 64  				# number of generator feature maps
	ndf = 64  				# number of discriminator feature maps
	num_epochs = 5  		# number of training epochs to run
	learning_rate = 0.0002	# for optimizers
	beta1 = 0.5  			# for adam optimizer
	ngpu = 0  				# numbers of gpu available, 0 for cpu mode
	
	# for model hyperparameters
	init_var = 0.02  		# variance for the initialization Gaussian
	slope = 0.2 			# slope for leaky ReLU
	sigma = 1 				# used in inference P(x|z=mu+sigma^1/2),
							# where P is a gaussian with variance sigma * I

	output_path = "results/v{}-seed{}-epoch{}/".format(version, manualSeed, num_epochs)
	model_output = output_path + "model.weights/"
	dataroot = "~/vision_datasets/celebA"
