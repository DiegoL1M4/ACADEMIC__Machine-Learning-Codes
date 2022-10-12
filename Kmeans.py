import random
import numpy as np
import matplotlib.pyplot as plt

from math import floor
from General import General

class Kmeans:   
    def train(K_value, data, percentage, movements):
        # Data division
        total = floor(len(data) * percentage)
        dataTrained = []
        dataToPredict = []

        counter = 1
        for sample in data:
            sample.append('')
            if(counter <= total):
                dataTrained.append(sample)
            else:
                dataToPredict.append(sample)
            counter += 1

        # Initial centroids selection
        centroids = []

        centroidsIndex = []
        for k in range(K_value):
            number = random.randrange(0, len(dataTrained))
            while (centroidsIndex != [] and number in centroidsIndex):
                number = random.randrange(0, len(dataTrained))
            
            centroids.append(dataTrained[number][0])
            centroidsIndex.append(number)

        # Movement cycle
        for k in range(movements):
            # Definition of Centroids
            for sample in dataTrained:
                distance = ''
                for centroid in centroids:
                    distanceCalc = General.calcDistance(centroid, sample[0])
                    if(distance == '' or distanceCalc < distance):
                        distance = distanceCalc
                        sample[2] = centroids.index(centroid)

            # Calc new centroids
            collection = []
            dataTrained = sorted(dataTrained, key=lambda case: case[2])
            typing = dataTrained[0][2]

            if(k != movements - 1):
                for coords in dataTrained:
                    if(typing != coords[2]):
                        centroids[typing] = np.mean(collection, axis=0).tolist()
                        typing = coords[2]
                        collection = []
                    collection.append(coords[0])
                centroids[typing] = np.mean(collection, axis=0).tolist()

    
        # Type definition
        nameTypes = []
        collection = []
        dataTypeGroup = []
        typing = dataTrained[0][2]
        for sample in dataTrained:
            if (sample[1] not in nameTypes):
                nameTypes.append(sample[1])

            if(typing != sample[2]):
                dataTypeGroup.append(collection)
                collection = []
                typing = sample[2]
            
            collection.append(sample)
        dataTypeGroup.append(collection)

        typeSum = []
        for k in nameTypes:
            typeSum.append(0)

        i = 0
        for centroidGroup in dataTypeGroup:
            for k in range(len(typeSum)):
                typeSum[k] = 0

            for group in centroidGroup:
                index = nameTypes.index(group[1])
                typeSum[index] += 1

            typing = ''
            selectedTypeSum = 0
            for k in range(len(nameTypes)):
                if(typing == '' or typeSum[k] > selectedTypeSum):
                    typing = nameTypes[k]
                    selectedTypeSum = typeSum[k]

            centroids[i] = [centroids[i], typing]
            i += 1
        
        return [centroids, dataToPredict]

    def predict(dataBase, sample):
        distance = ''
        result = ''
        for test in dataBase:
            distanceCalc = General.calcDistance(test[0], sample)
            if(distance == '' or distanceCalc < distance):
                distance = distanceCalc
                result = test[1]
            
        return result
