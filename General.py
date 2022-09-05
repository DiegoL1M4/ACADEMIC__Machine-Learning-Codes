import math
import random
import os
from this import d
from matplotlib.pyplot import axis
import numpy as np
import seaborn as sns

from ntpath import join

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

class General:
    def getData(dataBaseName):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        file = open(join(current_dir, 'data/' + dataBaseName + '.data'), 'r')
        data = []
        coordsData = []

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

            coordsData.append(sample[0])
            data.append(sample)
            row = file.readline()

        # Normalization
        minArray = np.matrix(coordsData).min(axis=0)
        maxArray = np.matrix(coordsData).max(axis=0)
        for k in range(len(data[0][0])):
            min = minArray.item(k)
            max = maxArray.item(k)
            print()

        for sample in data:
            for k in range(len(sample[0])):
                sample[0][k] = (sample[0][k] - minArray.item(k)) / (maxArray.item(k) - minArray.item(k))

        # Shuffle
        random.shuffle(data)
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
        heatmap.set_title('Matriz de Confusão\n')
        heatmap.set_xlabel('Valores Previstos')
        heatmap.set_ylabel('Valores Reais')

        plt.savefig("ConfusionMatrix " + str(order + 1) + ".png")
        plt.clf()

    def twoCoordsData(data):
        newBase = []
        for k in data:
            newBase.append([k[0][0], k[0][1], k[1]])
        return newBase

    def plotDecisionSurface(decisionData):
        scatter = sns.scatterplot(x="A", y="B", hue="Class", data=decisionData)
        scatter.set_title('Superfície de Decisão\n')
        plt.show()

        return 0
    