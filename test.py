
from General import General
from KNN import KNN
from DMC import DMC
from Kmeans import Kmeans
from DecisionSurface import DecisionSurface

# Variables
dataBasePercentage = 0.8
totalExec = 20

K_value_KNN = 5
K_value_Kmeans = 20
movements = 100
algorithm = "Kmeans" # KNN | DMC | Kmeans
dataBaseName = "artificial1" # iris | column | artificial1

# Operation 
hitRateList = []
decisionSurface = DecisionSurface()

for k in range(totalExec):
    data = General.normalization(General.getData(dataBaseName))

    if(algorithm == "KNN"):
        dataPart = KNN.train(data, dataBasePercentage)
        predictResult = KNN.predict(K_value_KNN, dataPart[0], dataPart[1])
    elif(algorithm == "DMC"):
        dataPart = DMC.train(data, dataBasePercentage)
        predictResult = DMC.predict(dataPart[0], dataPart[1])
    elif(algorithm == "Kmeans"):
        predictResult = []
        dataPart = Kmeans.train(K_value_Kmeans, data, dataBasePercentage, movements)

        for sample in dataPart[1]:
            predictResult.append(Kmeans.predict(dataPart[0], sample))

    decisionSurface.plot(algorithm, General.twoCoordsData(dataPart[0], 'list'), General.twoCoordsData(data, 'dataFrame'))
    
    print("\nPredição " + str(k + 1) + ":")
    print(predictResult)
    
    General.plotConfusionMatrix(General.confusionMatrix(dataPart[0], predictResult), k)

    hitRateList.append(General.hitRate(predictResult))
    

print("\nLista das taxas de acerto:")
print(hitRateList)

print("\n" + algorithm + " - " + dataBaseName)
print("Acurácia: " + str(General.average(hitRateList)))
print("Desvio Padrão: " + str(General.standardDeviation(hitRateList)))
