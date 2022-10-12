import numpy as np

from math import floor
from General import General

class NaiveBayes:
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
                dataTrained.append([[np.mean(collection, axis=0), np.std(collection, axis=0)], typing, len(collection) / total])
                typing = coords[1]
                collection = []

            collection.append(coords[0])
        dataTrained.append([[np.mean(collection, axis=0), np.std(collection, axis=0)], typing, len(collection) / total])

        return [dataTrained, dataToPredict]
            
    def predict(dataBase, sample):
        result = ''
        probCalc = 0

        for test in dataBase:
            prob = General.PDF(sample, test)

            if(probCalc < prob * test[2]):
                probCalc = prob
                result = test[1]
            
        return result
