
# CUDA settings
# use_cuda = torch.cuda.is_available()

# Data set parameters
dataName = 'BSDS500'
trainDataPath = './datasets/images/train'
imgSize = [321, 481]

# Initialization parameters
K = 4

# DSCN paramters
# num_sample = x.shape[0]
channels = [1, 5]
kernels = [3]
num_epoch = 500
weight_coef = 1.0
weight_selfExp = 75
learning_rate = 0.001

# post clustering parameters
alapha = 0.4 # threshold of C
dim_subspace = 3 #dimension of each subspace
ro = 8

# save path
savePath = './DSCNet/' + dataName + '.pt'