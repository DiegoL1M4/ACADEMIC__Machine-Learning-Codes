
import math
import random
import scipy
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

class Utils:
    def normalization(data):
        coordsData = []
        for sample in data:
            coordsData.append(sample[0])

        minArray = np.matrix(coordsData).min(axis=0)
        maxArray = np.matrix(coordsData).max(axis=0)

        for sample in data:
            for k in range(len(sample[0])):
                sample[0][k] = (sample[0][k] - minArray.item(k)) / (maxArray.item(k) - minArray.item(k))

        return data

    def min_max_normalization(X):
        X_norm = X.copy()

        for column in X_norm.columns:
            X_norm[column] = (X_norm[column] - np.min(X_norm[column])) / (np.max(X_norm[column]) - np.min(X_norm[column]))

        return X_norm

    def plotDistribution(column, feature_name, distr_name):
        distribution = getattr(scipy.stats, distr_name)
        distri_params = distribution.fit(column)
        new_distr = distribution(*distri_params)

        x = np.linspace(0, 1, 100)
        _, ax = plt.subplots(1, 2)

        plt.suptitle(feature_name + ' - ' + distr_name)
        sns.histplot(x=column, ax=ax[0])
        sns.lineplot(x=x, y=new_distr.pdf(x), ax=ax[1])
        plt.show()

    def hitRate(dataList):
        hits = 0
        for target in dataList:
            if(target[1] == target[2]):
                hits += 1
        return (hits / len(dataList))

    def average(values):
        total = 0
        for k in values:
            total += k
        return total / len(values)

    def standardDeviation(values):
        total = 0
        average = Utils.average(values)
        for value in values:
            total += math.pow(value - average, 2)
        return math.sqrt(total / len(values))

    def isNumber(value):
        try:
            float(value)
        except ValueError:
            return False
        return True

    def copyMatrix(data, type):
        newData = []
        for k in data:
            newData.append(k[:])

        if(type == 1):
            return newData
        else:
            return pd.DataFrame(newData, columns=['A', 'B', 'C', 'D'])

    def calcDistance(point1, point2):
        total = 0
        for coord in range(len(point1)):
            if(point1[coord] != "?" and point2[coord] != "?"):
                total += math.pow(point2[coord] - point1[coord], 2)
        return math.sqrt(total)

    def getDataBase(dataBaseName):
        file = open('data/' + dataBaseName + '.data', 'r')
        data = []

        # Read file
        row = file.readline()
        while(row != '' and row != '\n'):
            sample = [[], '']
            elements = row.replace('\n', '')
            
            # Select divider type
            elements = elements.split(',')
            if(len(elements) < 2):
                elements = elements[0].split(' ')

            # Get data
            for e in elements:
                if(Utils.isNumber(e)):
                    sample[0].append(float(e)) #coord
                else:
                    sample[1] = e #type
            
            data.append(sample)
            row = file.readline()
        file.close()

        random.shuffle(data)
        return data

    def getDataBasePandas(dataBaseName):
        file = open('data/' + dataBaseName + '.data', 'r')
        data = []

        # Read file
        row = file.readline()
        while(row != '' and row != '\n'):
            sample = []
            elements = row.replace('\n', '')
            
            # Select divider type
            elements = elements.split(',')
            if(len(elements) < 2):
                elements = elements[0].split(' ')

            # Get data
            for e in elements:
                if(Utils.isNumber(e)):
                    sample.append(float(e)) #coord
                else:
                    sample.append(e) #type

            data.append(sample)
            row = file.readline()
        file.close()

        if(dataBaseName == "iris"):
            columns = [
                'sepal_length',
                'sepal_width', 
                'petal_length', 
                'petal_width', 
                'class'
            ]
        else:
            columns = [
                'pelvic_incidence', 
                'pelvic_tilt', 
                'lumbar_lordosis_angle', 
                'sacral_slope', 
                'pelvic_radius', 
                'degree_spondylolisthesis',
                'class'
            ]

        data = pd.DataFrame(data, columns=columns)
        data = Utils.min_max_normalization(data.drop(data.columns[-1:], axis=1))

        return data

    def getCoordsMatrix(dataBaseName):
        data = Utils.getDataBase(dataBaseName)
        newData = []

        for k in data:
            newData.append(k[0])

        return newData

    def RMSE(data, dataAdj):
        sub = np.subtract(data, dataAdj)
        mse = np.square(sub).mean()

        return math.sqrt(mse)
