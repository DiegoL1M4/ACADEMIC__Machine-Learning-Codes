from General import General
from KNN import KNN
from DMC import DMC
from Kmeans import Kmeans
from DecisionSurface import DecisionSurface

# Execution Variables
dataBasePercentage = 0.8
totalExec = 20

K_value_KNN = 6
K_value_Kmeans = 20
movements = 100

onlyGenerateDataFiles = False
algorithm = "KNN" # KNN | DMC | Kmeans
dataBaseName = "iris" # iris | column | artificial1

# Operation Variables
hitRateList = []
decisionSurface = DecisionSurface()

# Generate shuffle bases
if(onlyGenerateDataFiles):
    for name in ["iris", "column", "artificial1"]:
        dataFile = General.readFile(name)
        for k in range(totalExec):
            General.newBaseShuffled(dataFile, name, k)

# Operation
else:
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

        print("\nTrained Data " + str(k + 1) + ":")
        print(dataPart[0])
        print("Predict Data " + str(k + 1) + ":")
        print(dataPart[1])

        # Test
        for sample in dataPart[1]:
            if(algorithm == "KNN"):
                sample[2] = KNN.predict(K_value_KNN, dataPart[0], sample[0])
            elif(algorithm == "DMC"):
                sample[2] = DMC.predict(dataPart[0], sample[0])
            elif(algorithm == "Kmeans"):
                sample[2] = Kmeans.predict(dataPart[0], sample[0])
            
            predictResult.append(sample)

        # Decision Surface and Confusion Matrix
        decisionSurface.plot(algorithm, General.twoCoordsData(dataPart[0], 'list'), General.twoCoordsData(data, 'dataFrame'), K_value_KNN)
        General.plotConfusionMatrix(General.confusionMatrix(dataPart[0], predictResult), k)
        
        # Result
        print("\nPredição " + str(k + 1) + ":")
        print(predictResult)

        hitRateList.append(General.hitRate(predictResult))
        
    # Final Results
    print("\nLista das taxas de acerto:")
    print(hitRateList)

    print("\n" + algorithm + " - " + dataBaseName)
    print("Acurácia: " + str(General.average(hitRateList)))
    print("Desvio Padrão: " + str(General.standardDeviation(hitRateList)))
