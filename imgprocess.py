# Functions of img processing.
import config
import numpy as np
import copy

def extractPixelBlock(originalImg, labels):
    '''
    input_param:
        originalImg: Original pixels matrix of input img. np.ndarray
        label: label matrix of input img. np.ndarray
    output_param:
        pixelBlockList: a list contains all pixelblock which incoporates same label pixels.
    '''
    # Copy a new labels due to max() function alter dimentions of its parameter
    newLabels = copy.deepcopy(labels)
    maxLabel = max(newLabels)
    pixelBlockList = []

    labels = labels.reshape(-1,1)

    blankBlock = np.array([255, 255, 255])
    for i in range(maxLabel + 1):
        # Uncomment line24 and comment line25 to visualize pixelBlock.
        # pixelBlock = [pixel if label == i else blankBlock for pixel, label in zip(originalImg, labels)]
        pixelBlock = [pixel if label == i else -1 for pixel, label in zip(originalImg, labels)]
        pixelBlock = np.array(pixelBlock)
        pixelBlock = pixelBlock.reshape(config.imgSize[0], config.imgSize[1], -1)
        pixelBlockList.append(pixelBlock)

    # pixelBlockList = np.array(pixelBlockList)

    return pixelBlockList

def extractFeature(elementlist):
    '''
    input_param:
        elementlist: A list contains all element.
    output_param:
        featurelist: A list contains each element's feature.
    '''
    pass