import math
import random
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ntpath import join

class General:
    def getData(dataBaseName):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        file = open(join(current_dir, 'data/' + dataBaseName + '.data'), 'r')
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

        # Shuffle
        random.shuffle(data)
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

        plt.savefig("graphics/ConfusionMatrix " + str(order + 1) + ".png")
        plt.clf()

    def twoCoordsData(data):
        newBase = []
        for k in data:
            newBase.append([k[0][0], k[0][1], k[1]])
        return newBase
        # return pd.DataFrame(np.array(newBase), columns=['A', 'B', 'Class'])

    def plotDecisionSurface(decisionData):
        # for k in decisionData:
        #     if(k[2] == 'A'):
        #         color = 'blue'
        #     else:
        #         color = 'red'
        #     plt.scatter(float(k[0]), float(k[1]), c = color)

        # plt.axis([-0.05, 1.05, -0.05, 1.05])
        # plt.grid(True)
        # plt.savefig('Artificial1.png')
        # plt.show()

        data = X_test_2d[:][:].values

        x_min, x_max = data[:, 0].min() - 0.1, data[:,0].max() + 0.1
        y_min, y_max = data[:, 1].min() - 0.1, data[:, 1].max() + 0.1

        xx, yy = np.meshgrid(np.linspace(x_min,x_max, 100),
        np.linspace(y_min, y_max, 100))

        x_in = np.c_[xx.ravel(), yy.ravel()]

        y_pred = [[dmc_model.predict(x) for x in x_in]]
        for i, y in enumerate(y_pred[0]):
            if y == 'Iris-setosa':
                y_pred[0][i] = 1
            elif y == 'Iris-virginica':
                y_pred[0][i] = 2
            else:
                y_pred[0][i] = 3

        y_pred = np.round(y_pred).reshape(xx.shape)

        plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7 )

        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

        dmc_y_pred = np.array([dmc_model.predict(x) for x in X_test_2d.values])

        setosa = np.where(dmc_y_pred == 'Iris-setosa')
        virginica = np.where(dmc_y_pred == 'Iris-virginica')
        versicolor = np.where(dmc_y_pred == 'Iris-versicolor')

        plt.scatter(data[setosa, 0], data[setosa, 1],
                    color='red', marker='o', label='setosa')
        plt.scatter(data[versicolor, 0], data[versicolor, 1],
                    color='blue', marker='X', label='versicolor')
        plt.scatter(data[virginica, 0], data[virginica, 1],
                    color='green', marker='P', label='virginica')

        plt.show()

        return 0
    