
import scipy
import math
import numpy as np

from math import floor
from Utils import Utils
from sklearn.linear_model import LinearRegression

class Functions:
    def naiveTrain(data, percentage):
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
            
    def naivePredict(dataBase, sample):
        result = ''
        probCalc = 0

        for test in dataBase:
            likelihood = Functions.likelihood(sample, test)
            priori = test[2]

            if(result == '' or probCalc < (likelihood * priori)):
                probCalc = (likelihood * priori)
                result = test[1]
            
        return result

    def likelihood(sample, classTrain):
        result = 1
        for k in range(len(sample)):
            mean = classTrain[0][0][k]
            std = classTrain[0][1][k]

            if(std == 0):
                std = 1
            
            total = 1 / (std * math.sqrt(2 * math.pi))
            exp = -(1/2) * math.pow((sample[k] - mean) / std , 2)

            result *= total * math.pow(math.e, exp)

        return result

    def CMI(dataAltered):
        for dataColumn in dataAltered.columns:
            base = dataAltered[dataAltered[dataColumn] != '?'][dataColumn].to_list()

            distribution = getattr(scipy.stats, 'norm')
            params = distribution.fit(base)
            distr = distribution(*params)
            
            dataAltered[dataColumn] = dataAltered[dataColumn].replace('?', distr.expect())
            
        return dataAltered.values

    def KNN(dataAltered, k_value):
        for sample in dataAltered:
            indexGroup = []
            for i, k in enumerate(sample):
                if(k == "?"):
                    indexGroup.append(i)

            if(len(indexGroup) == 0):
                continue

            for attributeIndex in indexGroup:
                base = []
                for sampleSearch in dataAltered:
                    count = 0
                    for k in sampleSearch:
                        if(k != "?"):
                            count += 1
                    if(sampleSearch[attributeIndex] != "?" and count > 2):
                        base.append(sampleSearch)
                
                samplesDistance = []
                for sampleBase in base:
                    distance = Utils.calcDistance(sample, sampleBase)
                    samplesDistance.append([sampleBase, distance])

                baseSorted = sorted(samplesDistance, key=lambda k: k[1])

                total = 0
                for k in range(k_value):
                    total += baseSorted[k][0][attributeIndex]

                sample[attributeIndex] = total/k_value

        return dataAltered

    def regression(dataAltered):
        for i, sample in enumerate(dataAltered.values):
            indexGroup = []
            columns = []
            columnsAltered = []
            values = np.array([])

            for j, k in enumerate(sample):
                if(k == "?"):
                    indexGroup.append(j)

            if(len(indexGroup) == 0):
                continue
            
            for j, k in enumerate(sample):
                if k != '?':
                    values = np.array([*values, k])
                    columns.append(dataAltered.columns[j])
                    continue

                columnsAltered.append(dataAltered.columns[j]) 

            regression = LinearRegression()
            regression.fit(
                dataAltered[ (dataAltered != '?').all(axis=1) ][columns].values,
                dataAltered[ (dataAltered != '?').all(axis=1) ][columnsAltered].values
            )

            dataAltered.loc[i][columnsAltered] = regression.predict([values])[0]

        return dataAltered.values
