
import seaborn as sns

from General import General
from KNN import KNN
from DMC import DMC
from Kmeans import Kmeans

# Variables
dataBasePercentage = 0.8
totalExec = 20

K_value = 20
movements = 200
algorithm = "Kmeans" # KNN | DMC | Kmeans
dataBaseName = "artificial1" # iris | column | artificial1


# Operation 
hitRateList = []
for k in range(totalExec):
    data = General.normalization(General.getData(dataBaseName))
    
    # General.plotDecisionSurface(General.twoCoordsData(data))

    if(algorithm == "KNN"):
        dataPart = KNN.train(data, dataBasePercentage)
        predictResult = KNN.predict(K_value, dataPart[0], dataPart[1])
    elif(algorithm == "DMC"):
        dataPart = DMC.train(data, dataBasePercentage)
        predictResult = DMC.predict(dataPart[0], dataPart[1])
    elif(algorithm == "Kmeans"):
        predictResult = []
        dataPart = Kmeans.train(K_value, data, dataBasePercentage, movements)

        for sample in dataPart[1]:
            predictResult.append(Kmeans.predict(dataPart[0], sample))


    print("\nPredição " + str(k + 1) + ":")
    print(predictResult)
    
    General.plotConfusionMatrix(General.confusionMatrix(dataPart[0], predictResult), k)

    hitRateList.append(General.hitRate(predictResult))

    newBase = General.twoCoordsData(predictResult)

print("\nLista das taxas de acerto:")
print(hitRateList)

print("\nAcurácia: " + str(General.average(hitRateList)))
print("Desvio Padrão: " + str(General.standardDeviation(hitRateList)))
