
from math import floor

import numpy as np

from General import General

class DMC:
    def train(data, percentage):
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
        
        return [dataTrained, dataToPredict]
            
    def predict(dataBase, dataPredict):
        dataMean = []
        dataBaseSorted = sorted(dataBase, key=lambda case: case[1])

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
        
        for sample in dataPredict:
            distance = ''
            for test in dataMean:
                distanceCalc = General.calcDistance(test[0], sample[0])
                if(distance == '' or distanceCalc < distance):
                    distance = distanceCalc
                    sample[2] = test[1]
            
        return dataPredict
