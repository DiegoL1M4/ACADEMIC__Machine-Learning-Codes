
import seaborn as sns

from General import General
from KNN import KNN
from DMC import DMC

# Variables
dataBasePercentage = 0.8
totalExe = 5

K_value = 4
algorithm = "KNN" # KNN | DMC
dataBaseName = "iris" # iris | column | artificial


# Operation 
hitRateList = []
for k in range(totalExe):
    data = General.getData(dataBaseName)

    if(algorithm == "KNN"):
        dataPart = KNN.train(data, dataBasePercentage)
        predictResult = KNN.predict(K_value, dataPart[0], dataPart[1])
    elif(algorithm == "DMC"):
        dataPart = DMC.train(data, dataBasePercentage)
        predictResult = DMC.predict(K_value, dataPart[0], dataPart[1])

    General.plotConfusionMatrix(General.confusionMatrix(dataPart[0], predictResult), k)

    print(predictResult)

    hitRateList.append(General.hitRate(predictResult))

    newBase = General.twoCoordsData(predictResult)

print(hitRateList)

print("Acurácia: " + str(General.average(hitRateList)))
print("Desvio Padrão: " + str(General.standardDeviation(hitRateList)))
