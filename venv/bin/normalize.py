import numpy as np
import cv2
import glob
import os
import pickle
import random

def sameFruit(currentFolderName, previousFolderName):
    if currentFolderName == previousFolderName:
        return True
    else:
        return False

def loadImages(directory):
    result = []
    folders = os.listdir(directory)
    folders.sort()
    previousFolderName = ""
    for folder in folders:
        imageNames = os.listdir(directory + folder + '/')
        isSame = sameFruit(folder.split(' ')[0], previousFolderName.split(' ')[0])
        if (not isSame):
            row = []
        for imageName in imageNames:
            path = directory + "/" + folder + "/" + imageName
            image = cv2.imread(path)
            row.append(image)
        if(not isSame):
            random.shuffle(row)
            result.append(row)
        previousFolderName = folder
    return result

def normalizeImages(images):
    result = []
    for imageRow in images:
        row = []
        for image in imageRow:
            normImage = cv2.normalize(image, None, alpha=-0.5, beta=0.5, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            row.append(normImage)
        result.append(row)
    return result

def pickleImages(images, fileName):
    outputFile = open(fileName, "wb")
    pickle.dump(images, outputFile)
    outputFile.close()

def unpickleImages(filename):
    openFile = open(filename, "rb")
    result = pickle.load(openFile)
    openFile.close()
    return result

def pickleImagesIntoSeparetPickles(images, saveDir, imgDir):
    folders = os.listdir(imgDir)
    folders.sort()
    index = 0
    previousFolderName = ""
    for folder in folders:
        isSame = sameFruit(folder.split(' ')[0], previousFolderName.split(' ')[0])
        if(not isSame):
            pickleImages(images[index], saveDir + folder.split(' ')[0])
            index += 1
        previousFolderName = folder

def loadAndNormalizeImages(path):
    images = loadImages(path)
    showImages(images)
    return normalizeImages(images)

def trainingAndValidationData(trainSetLength, validateSetLength, data):
    trainingData = []
    valdiationData = []
    for row in data:
        trainingData.append(row[:trainSetLength])
        valdiationData.append(row[trainSetLength:trainSetLength + validateSetLength])
    pickleImagesIntoSeparetPickles(trainingData, "Training/", "../Training/")
    pickleImagesIntoBundle(trainingData, "Training/Bundle/", "TrainingBundle")
    pickleImagesIntoSeparetPickles(valdiationData, "Validation/", "../Training/")
    pickleImagesIntoBundle(valdiationData, "Validation/Bundle/", "ValidationBundle")

def testingData(testingSetLength, data):
    testingData = []
    for row in data:
        testingData.append(row[:testingSetLength])
    pickleImagesIntoSeparetPickles(testingData, "Testing/", "../Training")
    pickleImagesIntoBundle(testingData, "Testing/Bundle/", "TestingBundle")

def pickleImagesIntoBundle(data, saveDir, filename):
    pickleImages(data, saveDir + filename)

def showImages(images):
    for row in images:
        i = 4;
        while i > 0:
            cv2.imshow('img', row[i])
            cv2.waitKey(0)
            i -= 1

if __name__ == "__main__":
    trainSetLength = 352
    testingSetLength = 70
    validateSetLength = 88
    random.randint(0,9)
    imgDirTrain = "../Training/"
    imgDirTest = "../Test/"
    trainImages = loadAndNormalizeImages(imgDirTrain)
    showImages(trainImages)
    testImages = loadAndNormalizeImages(imgDirTest)

    trainingAndValidationData(trainSetLength, validateSetLength, trainImages)
    testingData(testingSetLength, testImages)

