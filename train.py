import torch
from torch._C import dtype
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 

import numpy as np 

from kmeans_pytorch import kmeans
import imgprocess
import utilits

import matplotlib.pyplot as plt

import config
import dscn

# Load img
img_path = config.trainDataPath
fileList = utilits.getAllName(img_path)
# It is a little experiment, hence I only use one image.
file = fileList[0]
img = plt.imread(file)


imgWriteable = np.array(img)

imgWriteable = imgWriteable.reshape(-1, 3)
imgTensor = torch.from_numpy(imgWriteable)

labels, clusterCenters = kmeans(X= imgTensor, num_clusters= config.K, distance='euclidean', device=torch.device('cuda:0'))
labels = np.array(labels)

pixelBlockList = imgprocess.extractPixelBlock(imgWriteable, labels)
featureList = imgprocess.extractFeature(pixelBlockList)

assert len(pixelBlockList) == len(featureList)

num_sample = len(featureList)

x = torch.tensor(featureList, dtype = torch.float32)
x = x.cuda() if config.use_cuda else x 

K = 2
DSCN = dscn.DSCNet(config.channels, num_sample, pixelBlockList)
if config.use_cuda:
    DSCN = DSCN.cuda()

optimizer = optim.Adam(DSCN.parameters(), lr= config.learning_rate)

for epoch in range(config.num_epoch):
    x, x_recon, z, z_recon = DSCN(x)
    loss = DSCN.loss_fn(x, x_recon, z, z_recon, weight_coef= config.weight_coef, weight_selfExp= config.weight_selfExp)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % config.show_frequence == 0 or epoch == config.num_epoch -1:
        C = DSCN.self_expression.Coefficient.detach()
        if config.use_cuda:
            C = C.cpu()
        C = C.numpy()
        
        y_pred = utilits.spectral_clustering(C, K, config.dim_subspace, config.alapha, config.ro)
        y_pred = y_pred.squeeze()
        y = y.squeeze()
        # print('Epoch {:02d}: loss={:.4f}, acc={:.4f}, nmi={:.4f}'.format(epoch, loss.item() / y_pred.shape[0], acc(y, y_pred), nmi(y, y_pred)))
        print('Epoch {:02d}: loss={:.4f}'.format(epoch, loss.item() / y_pred.shape[0]))


torch.save(DSCN, config.save_path)