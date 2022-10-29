import numpy as np

from math import floor
from utils.General import General

class DMC:
    def train(data, percentage):
        # Data division
        total = floor(len(data) * percentage)
        dataTrained = []
        dataToPredict = []

        counter = 1
        for sample in data:
            if(counter <= total):
                dataTrained.append(sample)
            else:
                sample.append('')
                dataToPredict.append(sample)
            counter += 1

        # Definition of centroids
        dataMean = []
        dataBaseSorted = sorted(dataTrained, key=lambda case: case[1])
        typing = ''
        collection = []

        for coords in dataBaseSorted:
            if(typing == ''):
                typing = coords[1]
            if(typing != coords[1]):
                dataMean.append([np.mean(collection, axis=0), typing])
                typing = coords[1]
                collection = []
            collection.append(coords[0])
        dataMean.append([np.mean(collection, axis=0), typing])
        
        return [dataMean, dataToPredict]
            
    def predict(dataBase, sample):
        distance = ''
        result = ''
        for test in dataBase:
            distanceCalc = General.calcDistance(test[0], sample)
            if(distance == '' or distanceCalc < distance):
                distance = distanceCalc
                result = test[1]
            
        return result
