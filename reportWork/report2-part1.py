import sys, os
import pandas as pd

if not sys.path[0] == os.path.abspath('..'):
    sys.path.insert(0, os.path.abspath('..'))

from Utils import Utils
from Functions import Functions

# Execution Variables
dataBasePercentage = 0.8
totalExec = 20

confusionMatrixActive = False # True | False

dataBaseName = "column" # iris | column
distributions = ['laplace', 'norm', 'gamma', 'lognorm', 'norm', 'lognorm']

hitRateList = []

# Column Distributions
data = Utils.getDataBasePandas(dataBaseName)

for i, feature in enumerate(data.columns):
    Utils.plotDistribution(data[feature], feature, distributions[i])

# Predict Execution
for k in range(totalExec):
    # Data
    data = Utils.normalization(Utils.getDataBase(dataBaseName))
    predictResult = []

    # Train
    dataPart = Functions.naiveTrain(data, dataBasePercentage)
   
    # Test
    for sample in dataPart[1]:
        sample[2] = Functions.naivePredict(dataPart[0], sample[0])
        
        predictResult.append(sample)

    # Confusion Matrix
    # if(confusionMatrixActive):
    #     General.plotConfusionMatrix(General.confusionMatrix(dataPart[0], predictResult), k)

    hitRateList.append(Utils.hitRate(predictResult))
    
# Final Results
print("\nLista das taxas de acerto:")
print(hitRateList)

print("Acurácia: " + str(round(Utils.average(hitRateList), 4)))
print("Desvio Padrão: " + str(round(Utils.standardDeviation(hitRateList), 4)))
