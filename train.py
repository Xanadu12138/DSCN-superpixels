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
import cv2

import config
import dscn

from skimage.segmentation import mark_boundaries
from skimage.segmentation._slic import _enforce_label_connectivity_cython



# Load img
img_path = config.trainDataPath
fileList = utilits.getAllName(img_path)
# It is a little experiment, hence I only use one image.
file = fileList[0]
img = plt.imread(file)
imgCV = cv2.imread(file)



imgWriteable = np.array(img)

imgWriteable = imgWriteable.reshape(-1, 3)
imgTensor = torch.from_numpy(imgWriteable)

labels, clusterCenters = kmeans(X= imgTensor, num_clusters= config.K, distance='euclidean', device=torch.device('cuda:0'))
imgTensor = imgTensor.view((config.imgSize[0],config.imgSize[1], 3))

colorFeatureList = imgprocess.regionColorFeatures(imgTensor, labels)
textureFeatureList = imgprocess.regionTextureFeatures(imgCV, labels)
edgeFeatureList = imgprocess.regionEdgeFeatures(imgCV, labels)
spatialFeatureList = imgprocess.regionSpatialFeatures(labels)
featureList = torch.cat((colorFeatureList, textureFeatureList, edgeFeatureList, spatialFeatureList), dim = 1)

num_sample = len(featureList)


featureList = featureList.cuda() if config.use_cuda else featureList


K = 200

DSCN = dscn.DSCNet(config.channels, num_sample)
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
        # print('Epoch {:02d}: loss={:.4f}, acc={:.4f}, nmi={:.4f}'.format(epoch, loss.item() / y_pred.shape[0], acc(y, y_pred), nmi(y, y_pred)))
        print('Epoch {:02d}: loss={:.4f}'.format(epoch, loss.item() / y_pred.shape[0]))


torch.save(DSCN, config.savePath)

# Predict phase

C = DSCN.self_expression.Coefficient.detach()
if config.use_cuda:
    C = C.cpu()
    x = x.cpu()

C = C.numpy()
x = x.numpy()
y_pred = utilits.spectral_clustering(C, K , config.dim_subspace, config.alapha, config.ro)

img = np.array(img)
reconLabel = utilits.reconLabel(y_pred, pixelBlockList)
reconLabel = reconLabel.reshape(1, 321, 481)
slic_result = _enforce_label_connectivity_cython(reconLabel.astype(np.int64), config.min_size, config.max_size)
slic_result = slic_result.squeeze()
reconLabel = reconLabel.squeeze()
marked = mark_boundaries(img, slic_result.astype(int), color=(1,0,0))

plt.subplot(141)
plt.imshow(img)
plt.subplot(142)
plt.imshow(reconLabel)
plt.subplot(143)
plt.imshow(slic_result)
plt.subplot(144)
plt.imshow(marked)
plt.show()