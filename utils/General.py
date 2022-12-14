import math
import random

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class General:
    def readFile(dataBaseName):
        file = open('data/' + dataBaseName + '.data', 'r')
        data = []

        # Read file
        row = file.readline()
        while(row != '' and row != '\n'):
            data.append(row)
            row = file.readline()
        file.close()

        return data

    def newBaseShuffled(data, dataBaseName, order):
        random.shuffle(data)

        file = open('dataShuffled/' + dataBaseName + str(order) + '.data', 'w')
        for k in data:
            file.write(k)
        file.close()

    def getData(dataBaseName, order):
        file = open('dataShuffled/' + dataBaseName + str(order) + '.data', 'r')
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

            for e in elements:
                if(General.isNumber(e)):
                    sample[0].append(float(e)) #coord
                else:
                    sample[1] = e #type

            data.append(sample)
            row = file.readline()

        return data

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

    def isNumber(value):
        try:
            float(value)
        except ValueError:
            return False
        return True

    def calcDistance(point1, point2):
        total = 0
        for coord in range(len(point1)):
            total += math.pow(point2[coord] - point1[coord], 2)
        return math.sqrt(total)

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
        average = General.average(values)
        for value in values:
            total += math.pow(value - average, 2)
        return math.sqrt(total / len(values))

    def PDF(sample, classTrain):
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
    
    def multivariateGaussian(sample, matrix, mean):
        determinant = np.linalg.det(matrix)

        if(determinant == 0):
            determinant = 1
            matrix = matrix + (np.eye(len(matrix)) * 1e-10)
        
        inverse = np.linalg.inv(matrix)
        
        total = 1 / ( math.pow(2 * math.pi, len(sample) / 2) * np.power(determinant, 1 / 2) )
        exp = -(1/2) * np.dot( np.dot((sample - mean), inverse) , (sample - mean))

        return total * math.pow(math.e, exp)

    def linearDiscriminante(sample, matrix, mean, p_ci):
        determinant = np.linalg.det(matrix)

        if(determinant == 0):
            determinant = 1
            matrix = matrix + (np.eye(len(matrix)) * 1e-10)
        
        inverse = np.linalg.inv(matrix)

        p1 = (1/2) * np.dot( np.dot(sample, inverse) , mean)
        p2 = (-1/2) * np.dot( np.dot(mean, inverse) , mean)
        p3 = (1/2) * np.dot( np.dot(mean, inverse) , sample)
        
        return p1 + p2 + p3 + math.log(p_ci, math.e)

    def quadraticDiscriminante(sample, matrix, mean, p_ci):
        determinant = np.linalg.det(matrix)

        if(determinant == 0):
            determinant = 1
            matrix = matrix + (np.eye(len(matrix)) * 1e-10)
        
        inverse = np.linalg.inv(matrix)

        p1 = (-1/2) * np.dot( np.dot(sample, inverse) , sample)
        p2 = (1/2) * np.dot( np.dot(sample, inverse) , mean)
        p3 = (1/2) * np.dot( np.dot(mean, inverse) , sample)
        p4 = (-1/2) * np.dot( np.dot(mean, inverse) , mean)
        
        return p1 + p2 + p3 + p4 + math.log(p_ci, math.e) + p_ci

    def confusionMatrix(dataTrain, dataPredict):
        legends = []
        matrix = []
        # Find legends
        for k in dataTrain:
            try:
                legends.index(k[1])
            except ValueError:
                legends.append(k[1])
        
        # Create matrix
        for k in legends:
            line = []
            for k in legends:
                line.append(0)
            matrix.append(line)

        # Form matrix
        for k in dataPredict:
            indexA = legends.index(k[1])
            indexB = legends.index(k[2])
            matrix[indexA][indexB] += 1

        return [legends, matrix]

    def plotConfusionMatrix(data, order):
        heatmap = sns.heatmap(data[1], annot=True, cmap='Blues', xticklabels=data[0], yticklabels=data[0])
        heatmap.set_title('Matriz de Confus??o\n')
        heatmap.set_xlabel('Valores Previstos')
        heatmap.set_ylabel('Valores Reais')

        plt.savefig("graphics/ConfusionMatrix " + str(order + 1) + ".png")
        plt.clf()

    def twoCoordsData(data, valueDecision1, valueDecision2, type):
        newBase = []
        for k in data:
            if(type == 'list'):
                newBase.append([[k[0][valueDecision1], k[0][valueDecision2]], k[1]])
            if(type == 'dataFrame'):
                newBase.append([k[0][valueDecision1], k[0][valueDecision2], k[1]])
        
        if(type == 'list'):
            return newBase
        if(type == 'dataFrame'):
            return pd.DataFrame(newBase, columns=['A', 'B', 'Class'])
    