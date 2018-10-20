from os import listdir
from os.path import isfile
import os
import glob
import numpy as np
import cv2
import random
import pickle
import math
import time


def loadPickleFile(pathDirectory,val):
    try:
        if(val):
            count = 0
            file = open(pathDirectory, "rb")
            result = pickle.load(file)
            name = file.name.split("/")
            label = name[len(name)-1]

            # while(count < 150 and count<len(result)):
            #     displayPicture(result[count], label)
            #     count = count + 1
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
        print(e.with_traceback())

def euclideanDistance(fruit):
    start = time.time()
    distance = []
    for i in range(0,len(fruit)-2):
        for j in range (i+1,len(fruit)-2):
            distance.append(np.linalg.norm(np.squeeze(fruit[i])-np.squeeze(fruit[j])))
    print(time.time() - start)
    return distance



fruit = loadPickleFile("/home/suns/Desktop/fruits-360/TrainData/Apple",True)

euclideanDistance(fruit)
