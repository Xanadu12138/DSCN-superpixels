# Functions of img processing.
from functools import total_ordering
import config
import numpy as np
import copy
import torch
import cv2

from skimage.color import rgb2gray
from XCSLBP import XCSLBP

def extractPixelBlock(originalImg, labels):
    '''
    input_param:
        originalImg: Original pixels matrix that squeezed to 2 dimentions of input img. np.ndarray
        labels: label matrix of input img. np.ndarray
    output_param:
        pixelBlockList: a list contains all pixelblock which incoporates same label pixels.
    '''
    # Copy a new labels due to max() function alters dimentions of its parameter
    newLabels = copy.deepcopy(labels)
    maxLabel = max(newLabels)
    pixelBlockList = []

    labels = labels.reshape(-1,1)

    blankBlock = np.array([255, 255, 255])
    for i in range(maxLabel + 1):
        # Uncomment line24 and comment line25 to visualize pixelBlock.
        # pixelBlock = [pixel if label == i else blankBlock for pixel, label in zip(originalImg, labels)]
        pixelBlock = [pixel if label == i else config.blankBlock for pixel, label in zip(originalImg, labels)]
        pixelBlock = np.array(pixelBlock)
        pixelBlock = pixelBlock.reshape(config.imgSize[0], config.imgSize[1], -1)
        pixelBlockList.append(pixelBlock)

    return pixelBlockList

def extractFeature(pixelBlockList):
    '''
    input_param:
        pixelBlockList: A list contains all element.
    output_param:
        featureList: A list contains each element's feature. feature contains 3 channel's mean value and mean position info.
    '''
    featureList = []
    
    for i in range(len(pixelBlockList)):
        pixelList = []
        locationList = []

        for y in range(len(pixelBlockList[0])):
            for x in range(len(pixelBlockList[1])):
                if (pixelBlockList[i][y][x] != config.blankBlock).any():
                    pixelList.append(list(pixelBlockList[i][y][x]))
                    locationList.append((x,y))

        colorFeature = np.mean(np.array(pixelList), axis=0)
        locationFeature = np.mean(np.array(locationList), axis=0)
        features = np.append(colorFeature, locationFeature)
        featureList.append(features)

    
    featureList = np.array(featureList)
    return featureList

# Optimized version
def regionColorFeatures(img, labels):
    '''
    input_param:
        img: img matrix. torch.tensor
        labels: Kmeans clustering labels. torch.tensor
    output_param:
        colorFeatureList: A list contains each element's feature. feature contains 3 channel's mean value.
    '''
    numlab = max(labels)
    rlabels = labels.view(config.imgSize)

    colorFeatureList = []
    
    grayFrame = torch.tensor(rgb2gray(img))
    redFrame = img[:, :, 0]
    greenFrame = img[:, :, 1]
    blueFrame = img[:, :, 2]
    for i in range(numlab + 1):
        f = torch.eq(rlabels, i)
        graySpLocal = torch.mean(grayFrame[f].float())
        redSpLocal = torch.mean(redFrame[f].float())
        greenSpLocal = torch.mean(greenFrame[f].float())
        blueSpLocal = torch.mean(blueFrame[f].float())
        colorFeature = [redSpLocal, greenSpLocal, blueSpLocal, graySpLocal]
        colorFeatureList.append(colorFeature)

    colorFeatureList = torch.tensor(colorFeatureList)

    return colorFeatureList
    
def regionTextureFeatures(img, labels):
    '''
    input_param:
        img: CV2.imread
        labels
    '''
    numlab = max(labels)
    rlabels = labels.view(config.imgSize)

    # I = rgb2gray(img)
    XCS = XCSLBP(img)

    XCS = XCS * (255/ 16)
    XCSframe = torch.tensor(XCS)
    
    textureFeatureList = []
    for i in range(numlab + 1):
        f = torch.eq(rlabels, i)
        XCSSpLocal = torch.mean(XCSframe[f].float())
        textureFeatureList.append(XCSSpLocal)

    textureFeatureList = torch.tensor(textureFeatureList)
    textureFeatureList = textureFeatureList.unsqueeze(1)

    return textureFeatureList
   
def regionEdgeFeatures(img, labels):
    '''
    input_param:
        img: CV2.imread
        labels
    '''
    numlab = max(labels)
    rlabels = labels.view(config.imgSize)

    # frame = rgb2gray(img)

    Gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    Gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)

    Gmag = np.sqrt(Gx**2.0 + Gy**2.0)
    Gdir = np.arctan2(Gy, Gx) * (180 / np.pi)

    Gx, Gy, Gmag, Gdir = torch.tensor(Gx), torch.tensor(Gy), torch.tensor(Gmag), torch.tensor(Gdir)

    edgeFeatureList = []

    for i in range(numlab + 1):
        f = torch.eq(rlabels, i)
        GxSpLocal = torch.mean(Gx[f].float())
        GySpLocal = torch.mean(Gy[f].float())
        GmagSpLocal = torch.mean(Gmag[f].float())
        GdirSpLocal = torch.mean(Gdir[f].float())
        edgeFeature = [GxSpLocal, GySpLocal, GmagSpLocal, GdirSpLocal]
        edgeFeatureList.append(edgeFeature)

    edgeFeatureList = torch.tensor(edgeFeatureList)

    return edgeFeatureList

def regionSpatialFeatures(labels):
    numlab = max(labels)
    rlabels = labels.view(config.imgSize)

    col, row = config.imgSize
    x = range(1, col + 1)
    y = range(1, row + 1)
    Sx, Sy = np.meshgrid(y, x)
    Sx, Sy = torch.tensor(Sx), torch.tensor(Sy)

    spatialFeatureList = []

    for i in range(numlab + 1):
        f = torch.eq(rlabels, i)
        SxSpLocal = torch.mean(Sx[f].float())
        SySpLocal = torch.mean(Sy[f].float())
        spatialFeature = [SxSpLocal, SySpLocal]
        spatialFeatureList.append(spatialFeature)

    spatialFeatureList = torch.tensor(spatialFeatureList)

    return spatialFeatureList