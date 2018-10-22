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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import DBSCAN



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
    start = time.time()
    print("start of KMeans")
    print("size ",len(fr))
    fruit = array(fr)
    X = fruit.reshape((-1, 3*100*100)).astype(np.float32)

    pca = PCA(n_components=3).fit(X)
    pca_2d = pca.transform(X)

    plt.figure("PLOT")
    plt.scatter(pca_2d[:,0], pca_2d[:,1])

    kmeans = KMeans(n_clusters=48, random_state=111)
    kmeans.fit(pca_2d)

    plt.figure("KMEANS")
    plt.scatter(pca_2d[:,0],pca_2d[:,1],c = kmeans.labels_)

    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=169, linewidths=30, color='pink', zorder=10)
    print("end of KMeans")
    print(time.time()-start)
    plt.show()



def countDBSCAN(fr):
    start = time.time()
    print("start of DBSCAN")
    print("size ",len(fr))

    fruit = array(fr)
    X = fruit.reshape((-1, 3*100*100)).astype(np.float32)

    pca = PCA(n_components=15).fit(X)
    pca_2d = pca.transform(X)

    plt.figure("PLOT")
    plt.scatter(pca_2d[:,0], pca_2d[:,1])

    dbscan = DBSCAN(eps=3, min_samples=2)
    dbscan.fit(pca_2d)

    plt.figure("DBSCAN")
    plt.scatter(pca_2d[:,0],pca_2d[:,1],c = dbscan.labels_)

    # centers = dbscan.core_sample_indices_
    # plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=169, linewidths=30, color='pink', zorder=10)
    print("end of DBSCAN")
    print(time.time()-start)
    plt.show()




# result = loadPickleFile("/home/suns/Desktop/fruits-360 (copy)/TestData/Apple", True)
result = loadPickleFile("/home/suns/Desktop/fruits-360 (copy)/TestData/Pickle_file", True)
#countKMeans(result)
countDBSCAN(result)
