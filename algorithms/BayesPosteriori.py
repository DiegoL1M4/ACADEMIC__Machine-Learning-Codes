import numpy as np

from math import floor
from utils.General import General

class BayesPosteriori:
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

        # Data calculation
        dataBaseSorted = sorted(dataTrained, key=lambda case: case[1])
        typing = dataBaseSorted[0][1]
        dataTrained = []
        collection = []

        for coords in dataBaseSorted:
            if(typing != coords[1]):
                dataTrained.append([[np.cov(collection, rowvar=False), np.mean(collection, axis=0)], typing, len(collection) / total])
                typing = coords[1]
                collection = []

            collection.append(coords[0])
        dataTrained.append([[np.cov(collection, rowvar=False), np.mean(collection, axis=0)], typing, len(collection) / total])

        return [dataTrained, dataToPredict]
            
    def predict(dataBase, sample):
        result = ''
        probCalc = 0

        for test in dataBase:
            covMatrix = test[0][0]
            mean = test[0][1]
            priori = test[2]

            likelihood = General.multivariateGaussian(sample, covMatrix, mean)

            if(result == '' or probCalc < (likelihood * priori)):
                probCalc = (likelihood * priori)
                result = test[1]
            
        return result
