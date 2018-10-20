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
            print("pickle file loaded"+pathDirectory)
            return result
        else:
            count = 0
            count2 = 0
            file = open(pathDirectory, "rb")
            result = pickle.load(file)
            print("pickle file loaded"+pathDirectory)
            return result

    except Exception as e:
        print(e)
        print(e.with_traceback())

def euclideanDistance(fruit):
    start = time.time()
    count = 0
    distance = []
    for i in range(0,len(fruit)-2):
        for j in range (i+1,len(fruit)-1):
            distance.append(np.linalg.norm(np.squeeze(fruit[i])-np.squeeze(fruit[j])))
            count += 1
    print(time.time() - start)
    return distance

path = "/home/suns/Desktop/fruits-360 (copy)/TestData/"
files = os.listdir(path)
for file in files:
    print(file)
    fruit = loadPickleFile(path+file,True)
    distance = euclideanDistance(fruit)
    with open(file+".txt", 'w') as f:
        for item in distance:
            f.write("%s\n" % item)


# distance = [1,2,3,5,6,5,7,2,4,6]
# n, bins, patches = plt.hist(x=d, bins='auto', color='#0504aa',
#                             alpha=0.7, rwidth=0.85)
# plt.grid(axis='y', alpha=0.75)
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('My Very Own Histogram')
# plt.text(23, 45, r'$\mu=15, b=3$')
# maxfreq = distance.max()
# # Set a clean upper y-axis limit.
# plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
