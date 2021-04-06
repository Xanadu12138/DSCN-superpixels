import torch

# CUDA settings
use_cuda = torch.cuda.is_available()

# Data set parameters
dataName = 'BSDS500'
trainDataPath = './datasets/images/train'
imgSize = [321, 481]

# Initialization parameters
K = 1000
# DSCN paramters
channels = [11, 22]

# num_sample = x.shape[0]
kernels = [3]
num_epoch = 500
weight_coef = 1.0
weight_selfExp = 75
learning_rate = 0.001
show_frequence = 10

# post clustering parameters
alapha = 0.4 # threshold of C
dim_subspace = 3 #dimension of each subspace
ro = 8

# SLIC connectivity parmeters
segment_size = 321 * 481 / 40
min_size = int(0.02 * segment_size)
max_size = int(3 * segment_size)

# imgprocess parameters
blankBlock = [-1, -1, -1]

# save path
savePath = './savedmoudle/' + dataName + '.pt'