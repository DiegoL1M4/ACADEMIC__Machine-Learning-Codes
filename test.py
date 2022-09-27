
import seaborn as sns

from General import General
from KNN import KNN
from DMC import DMC

# Variables
dataBasePercentage = 0.8
totalExe = 20

K_value = 5
algorithm = "KNN" # KNN | DMC
dataBaseName = "iris" # iris | column | artificial1


# Operation 
hitRateList = []
for k in range(totalExe):
    data = General.normalization(General.getData(dataBaseName))
    
    # General.plotDecisionSurface(General.twoCoordsData(data))

    if(algorithm == "KNN"):
        dataPart = KNN.train(data, dataBasePercentage)
        predictResult = KNN.predict(K_value, dataPart[0], dataPart[1])
    elif(algorithm == "DMC"):
        dataPart = DMC.train(data, dataBasePercentage)
        predictResult = DMC.predict(dataPart[0], dataPart[1])

    print("\nPredição " + str(k + 1) + ":")
    print(predictResult)
    
    General.plotConfusionMatrix(General.confusionMatrix(dataPart[0], predictResult), k)

    hitRateList.append(General.hitRate(predictResult))

    newBase = General.twoCoordsData(predictResult)

print("\nLista das taxas de acerto:")
print(hitRateList)

print("\nAcurácia: " + str(General.average(hitRateList)))
print("Desvio Padrão: " + str(General.standardDeviation(hitRateList)))
