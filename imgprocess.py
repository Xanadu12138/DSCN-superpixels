# Functions of img processing.
import config
import numpy as np
import copy

def extractPixelBlock(originalImg, labels):
    '''
    input_param:
        originalImg: Original pixels matrix that squeezed to 2 dimentions of input img. np.ndarray
        labels: label matrix of input img. np.ndarray
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
                if pixelBlockList[i][y][x] != -1:
                    pixelList.append(list(pixelBlockList[i][y][x]))
                    locationList.append((x,y))

        colorFeature = np.mean(np.array(pixelList), axis=0)
        locationFeature = np.mean(np.array(locationList), axis=0)
        features = np.append(colorFeature, locationFeature)
        featureList.append(features)

    return featureList
