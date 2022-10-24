
import pandas as pd

from KNN import KNN
from DMC import DMC
from Kmeans import Kmeans
from BayesPosteriori import BayesPosteriori
from NaiveBayes import NaiveBayes

from General import General
from DecisionSurface import DecisionSurface

# Execution Variables
decisionSurface = DecisionSurface()
dataBasePercentage = 0.8
totalExec = 20
generateDataFiles = False # True | False

K_value_KNN = 6
K_value_Kmeans = 20
movements = 100

valueDecision1 = 0
valueDecision2 = 1
decisionSurfaceActive = False # True | False
confusionMatrixActive = True # True | False

algorithm = "NaiveBayes" # KNN | DMC | Kmeans | BayesPosteriori | NaiveBayes
dataBaseName = "dermatologyMod" # iris | column | artificial1 | breastMod | dermatologyMod | artificialII

# Generate shuffle bases
if(generateDataFiles):
    # ["iris", "column", "breastMod", "dermatologyMod", "artificial1", "artificialII"]
    for name in ["artificialII"]:
        dataFile = General.readFile(name)
        for k in range(totalExec):
            General.newBaseShuffled(dataFile, name, k)

# Operation
# for K_value_KNN in range(4, 11):
# for K_value_Kmeans in [1, 3, 5, 10, 15, 20, 30]:
for unique in range(1):
    hitRateList = []

    for k in range(totalExec):
        # Data
        data = General.normalization(General.getData(dataBaseName, k))
        predictResult = []

        # Train
        if(algorithm == "KNN"):
            dataPart = KNN.train(data, dataBasePercentage)
        elif(algorithm == "DMC"):
            dataPart = DMC.train(data, dataBasePercentage)
        elif(algorithm == "Kmeans"):
            dataPart = Kmeans.train(K_value_Kmeans, data, dataBasePercentage, movements)
        elif(algorithm == "BayesPosteriori"):
            dataPart = BayesPosteriori.train(data, dataBasePercentage)
        elif(algorithm == "NaiveBayes"):
            dataPart = NaiveBayes.train(data, dataBasePercentage)

        # Test
        for sample in dataPart[1]:
            if(algorithm == "KNN"):
                sample[2] = KNN.predict(K_value_KNN, dataPart[0], sample[0])
            elif(algorithm == "DMC"):
                sample[2] = DMC.predict(dataPart[0], sample[0])
            elif(algorithm == "Kmeans"):
                sample[2] = Kmeans.predict(dataPart[0], sample[0])
            elif(algorithm == "BayesPosteriori"):
                sample[2] = BayesPosteriori.predict(dataPart[0], sample[0])
            elif(algorithm == "NaiveBayes"):
                sample[2] = NaiveBayes.predict(dataPart[0], sample[0])
            
            predictResult.append(sample)

        # Confusion Matrix
        if(confusionMatrixActive):
            General.plotConfusionMatrix(General.confusionMatrix(dataPart[0], predictResult), k)

        # Decision Surface
        if(decisionSurfaceActive):
            dataDecision = General.twoCoordsData(data, valueDecision1, valueDecision2, 'list')

            if(algorithm == "KNN"):
                dataPartDecision = KNN.train(dataDecision, dataBasePercentage)
            elif(algorithm == "DMC"):
                dataPartDecision = DMC.train(dataDecision, dataBasePercentage)
            elif(algorithm == "Kmeans"):
                dataPartDecision = Kmeans.train(K_value_Kmeans, dataDecision, dataBasePercentage, movements)
            elif(algorithm == "NaiveBayes"):
                dataPartDecision = NaiveBayes.train(dataDecision, dataBasePercentage)
            elif(algorithm == "BayesPosteriori"):
                dataPartDecision = BayesPosteriori.train(dataDecision, dataBasePercentage)

            decisionSurface.plot(algorithm, dataPartDecision[0], General.twoCoordsData(data, valueDecision1, valueDecision2, 'dataFrame'), K_value_KNN)

        # Train   
        # print("\nTrained Data " + str(k + 1) + ":")
        # print(dataPart[0])

        # Result
        # print("\nPredição " + str(k + 1) + ":")
        # print(predictResult)

        hitRateList.append(General.hitRate(predictResult))
        
    # Final Results
    print("\nLista das taxas de acerto:")
    print(hitRateList)

    print("\n" + algorithm + " - " + dataBaseName + " | KNN: " + str(K_value_KNN) + " - Kmeans: " + str(K_value_Kmeans))
    print("Acurácia: " + str(round(General.average(hitRateList), 4)))
    print("Desvio Padrão: " + str(round(General.standardDeviation(hitRateList), 4)))
