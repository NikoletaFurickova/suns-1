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
from sklearn.cluster import KMeans
from numpy  import array
from sklearn import cluster, datasets
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
#import matplotlib
#matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets.samples_generator import make_blobs



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
            distance.append(np.linalg.norm(np.squeeze(fruit[i])-np.squeeze(fruit[j]))) ##a = np.array() a.flatten
            count += 1
    print(time.time() - start)
    return distance


def createDataforTxt():
    path = "/home/suns/Desktop/fruits-360 (copy)/TestData/"
    files = os.listdir(path)
    for file in files:
        print(file)
        fruit = loadPickleFile(path+file,True)
        distance = euclideanDistance(fruit)
        with open(file+".txt", 'w') as f:
            for item in distance:
                f.write("%s\n" % item)

def displayPicture(img, label):
    try:
        cv2.imshow(label, img)
        cv2.waitKey(100)
        time.sleep(2)
    except Exception as e:
        print(e.with_traceback())

def countKMeans(fr):
    fruit = array(fr)
    # print(fruit)
    # fruit = np.squeeze(fruit)
    # fruit = np.squeeze(fruit)
    # a = fruit.size
    # b = len(fruit[0])
    # a = int(a/b)
    print("***************************************************")
    # print(fruit)
    #print(fruit[1])
    # kmeans = KMeans(n_clusters=2).fit(fruit.reshape(a,b))
    # with open("subor.txt") as f:
    #     for i in kmeans.labels_:
    #         print(i)
    #print(kmeans.cluster_centers_)
######DOBRE SKORO
    # X = fruit.reshape((-1, 3*100*100)).astype(np.float32)
    # kmeans = KMeans(n_clusters=48)
    # kmeans.fit(X)
    # y_kmeans = kmeans.predict(X)
    # #
    # # plt.scatter()
    #
    #
    # plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    # centers = kmeans.cluster_centers_
    # plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    # plt.show()
######RASTO
    X = fruit.reshape((-1, 3*100*100)).astype(np.float32)
    # kmeans = KMeans(n_clusters=48)
    # kmeans.fit(X)
    pca = PCA(n_components=3).fit(X)
    pca_2d = pca.transform(X)

    plt.figure("PLOT")
    plt.scatter(pca_2d[:,0], pca_2d[:,1])

    kmeans = KMeans(n_clusters=48, random_state=111)
    kmeans.fit(pca_2d)

    plt.figure("KMEANS")
    plt.scatter(pca_2d[:,0],pca_2d[:,1],c = kmeans.labels_)

    centers = kmeans.cluster_centers_
    pca = PCA().fit(X)
    plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=169, linewidths=30, color='pink', zorder=10)
    plt.show()





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

# result = loadPickleFile("/home/suns/Desktop/fruits-360 (copy)/TestData/Apple", True)
result = loadPickleFile("/home/suns/Desktop/fruits-360 (copy)/TestData/Pickle_file", True)
#displayPicture(result[3],"bla")
countKMeans(result)
