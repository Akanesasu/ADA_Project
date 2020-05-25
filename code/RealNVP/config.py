
class config(object):
	
	manualSeed = 999		# random seed for reproducibility
	version = 1.0			# version
	
	workers = 2				# number of worker threads for loading
	batch_size = 64			# as RealNVP paper
	image_size = 64			# images are nc x image_size x image_size
	nc = 3					# number of channels (3 for RGB)
	nf = 64					# number of coupling layer network feature maps
	nCouplingLayers = 4		# number of coupling layers (as in NICE paper)
	
	num_epochs = 5			# number of training epochs to run
	learning_rate = 0.001	# for optimizers
	l2_weight_decay = 5e-5	# for adam optimizer
	ngpu = 0				# numbers of gpu available, 0 for cpu mode
	
	# for model hyperparameters
	init_var = 0.02			# variance for the initialization Gaussian
	slope = 0.2				# slope for leaky ReLU
	
	output_path = "results/v{}-seed{}-epoch{}/".format(version, manualSeed, num_epochs)
	model_output = output_path + "model.weights/"
	dataroot = "~/vision_datasets/celebA"