import numpy as np
import matplotlib.pyplot as plt

from algorithms.DMC import DMC
from utils.General import General
from algorithms.KNN import KNN
from algorithms.Kmeans import Kmeans
from algorithms.BayesPosteriori import BayesPosteriori
from algorithms.NaiveBayes import NaiveBayes
from algorithms.LinearDiscriminant import LinearDiscriminant
from algorithms.QuadraticDiscriminant import QuadraticDiscriminant

class DecisionSurface:
    def __init__(self):
        self.order = 1

    def plot(self, algorithm, dataPart, decisionData, K_value_KNN):
        test = decisionData.drop(decisionData.columns[-1:], axis=1)
        data = test[:][:].values

        x_min, x_max = data[:, 0].min() - 0.1, data[:,0].max() + 0.1
        y_min, y_max = data[:, 1].min() - 0.1, data[:, 1].max() + 0.1

        xx, yy = np.meshgrid(np.linspace(x_min,x_max, 100),
        np.linspace(y_min, y_max, 100))

        x_in = np.c_[xx.ravel(), yy.ravel()]
        if(algorithm == "KNN"):
            y_pred = [[KNN.predict(K_value_KNN, dataPart, x) for x in x_in]]
        elif(algorithm == "DMC"):
            y_pred = [[DMC.predict(dataPart, x) for x in x_in]]
        elif(algorithm == "Kmeans"):
            y_pred = [[Kmeans.predict(dataPart, x) for x in x_in]]
        elif(algorithm == "NaiveBayes"):
            y_pred = [[NaiveBayes.predict(dataPart, x[:-1]) for x in x_in]]
        elif(algorithm == "BayesPosteriori"):
            y_pred = [[BayesPosteriori.predict(dataPart, x[:-1]) for x in x_in]]
        elif(algorithm == "LinearDiscr"):
            y_pred = [[LinearDiscriminant.predict(dataPart, x) for x in x_in]]
        elif(algorithm == "QuadraticDiscr"):
            y_pred = [[QuadraticDiscriminant.predict(dataPart, x) for x in x_in]]

        listNames = []
        for i, y in enumerate(y_pred[0]):
            if(y not in listNames):
                listNames.append(y)
            y_pred[0][i] = listNames.index(y) + 1

        y_pred = np.round(y_pred).reshape(xx.shape)

        plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7 )

        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

        if(algorithm == "KNN"):
            algorithm_pred = np.array([KNN.predict(K_value_KNN, dataPart, x) for x in decisionData.values])
        elif(algorithm == "DMC"):
            algorithm_pred = np.array([DMC.predict(dataPart, x) for x in decisionData.values])
        elif(algorithm == "Kmeans"):
            algorithm_pred = np.array([Kmeans.predict(dataPart, x) for x in decisionData.values])
        elif(algorithm == "NaiveBayes"):
            algorithm_pred = np.array([NaiveBayes.predict(dataPart, x[:-1]) for x in decisionData.values])
        elif(algorithm == "BayesPosteriori"):
            algorithm_pred = np.array([BayesPosteriori.predict(dataPart, x[:-1]) for x in decisionData.values])
        elif(algorithm == "LinearDiscr"):
            algorithm_pred = np.array([LinearDiscriminant.predict(dataPart, x[:-1]) for x in decisionData.values])
        elif(algorithm == "QuadraticDiscr"):
            algorithm_pred = np.array([QuadraticDiscriminant.predict(dataPart, x[:-1]) for x in decisionData.values])

        separateDataList = []
        for dataType in listNames:
            separateDataList.append(np.where(algorithm_pred == dataType))

        colors = ['red', 'blue', 'green']
        symbols = ['o', 'X', 'P']
        for i, dataGroup in enumerate(separateDataList):
            plt.scatter(data[dataGroup, 0], data[dataGroup, 1], color=colors[i], marker=symbols[i])

        plt.savefig("graphics/DecisionSurface " + str(self.order) + ".png")
        plt.clf()

        self.order += 1

        return 0