from os import listdir
from os.path import isfile
import os
import glob
import numpy as np
import cv2
import random
import pickle

labels = []


def loadImages(path):
    fruit = []
    last_label = " "
    dirs = sorted(os.listdir(path))
    row = []
    for folder in dirs:
        fruit_name = folder.split(' ',1)[0]
        if (last_label == " "):
            last_label = fruit_name
        if((last_label != fruit_name) and (row != [])):
            labels.append(last_label)
            last_label = fruit_name
            fruit.append(row)
            row = []
        name = os.listdir(path+folder)
        count = 0
        for file in name:
            full_fruit_path = path+folder+"/"+file
            img = ((cv2.imread(full_fruit_path)))
            #if (count < 2):
                #displayPicture(img, last_label)
            img = cv2.normalize(img, None, alpha = 0, beta = 1, norm_type= cv2.NORM_MINMAX, dtype= cv2.CV_32F)
            row.append(img)
            #displayPicture(row[len(row)-1],"bla")
            if (count < 2):
                #displayPicture(img, last_label)
                count = count + 1
        cv2.destroyAllWindows()

    labels.append(last_label)
    fruit.append(row)
    return fruit

def divideTrainingFiles(training_fruit):
    training_set = []
    validation_set = []
    min = len(training_fruit[0])

    for fruit in training_fruit:
        random.shuffle(fruit)
        if (len(fruit) < min):
            min = len(fruit)

    for fruit in training_fruit:
        training_set.append(fruit[:(int(min*0.8))])
        validation_set.append(fruit[((-int(min*0.2)))-1:])
    return training_set, validation_set


def createPickleFile(pathDirectory, fruit_set, val):
    try:
        if not os.path.exists(pathDirectory):
            os.makedirs(pathDirectory)
        if (val):
            i = 0
            for fruit in fruit_set:
                pickle_file = open(pathDirectory+"/"+labels[i], "wb")
                pickle.dump(fruit, pickle_file)
                pickle_file.close()
                i = i + 1
        else:
            pickle_file = open(pathDirectory+"/Pickle_file", "wb")
            pickle.dump(fruit_set, pickle_file)
            pickle_file.close()

    except Exception as e:
        print(e)


def loadPickleFile(pathDirectory,val):
    try:
        if(val):
            count = 0
            file = open(pathDirectory, "rb")
            result = pickle.load(file)
            name = file.name.split("/")
            label = name[len(name)-1]

            while(count < 150 and count<len(result)):
                displayPicture(result[count], label)
                count = count + 1

            file.close()
            cv2.destroyAllWindows()
            print("pickle file loaded")
            return result
        else:
            count = 0
            count2 = 0
            file = open(pathDirectory, "rb")
            result = pickle.load(file)
            print("pickle file loaded")
            return result

    except Exception as e:
        print(e)

def displayPicture(img, label):
    try:
        cv2.imshow(label, img)
        cv2.waitKey(100)
    except Exception as e:
        print(e)




# training_fruit = loadImages("/home/suns/Desktop/fruits-360 (copy)/Training/")
# testing_set = loadImages("/home/suns/Desktop/fruits-360 (copy)/Test/")
# training_set, validation_set = divideTrainingFiles(training_fruit)
# createPickleFile("/home/suns/Desktop/fruits-360 (copy)/TrainData", training_set, True)
# createPickleFile("/home/suns/Desktop/fruits-360 (copy)/TestData", testing_set, True)
# createPickleFile("/home/suns/Desktop/fruits-360 (copy)/ValidationData", validation_set, True)
# createPickleFile("/home/suns/Desktop/fruits-360 (copy)/TrainData", training_set, False)
# createPickleFile("/home/suns/Desktop/fruits-360 (copy)/TestData", testing_set, False)
# createPickleFile("/home/suns/Desktop/fruits-360 (copy)/ValidationData", validation_set, False)
loadPickleFile("/home/suns/Desktop/fruits-360 (copy)/TrainData/Apple",True)



# training_fruit = loadImages("Training")
# testing_set = loadImages("Test")
# training_set, validation_set = divideTrainingFiles(training_fruit)
# createPickleFile("/home/suns/Desktop/fruits-360/TrainData", training_set)
# createPickleFile("/home/suns/Desktop/fruits-360/TestData", testing_set)
# createPickleFile("/home/suns/Desktop/fruits-360/ValidationData", validation_set)
# loadPickleFile("/home/suns/Desktop/fruits-360/TrainData/Banana")
