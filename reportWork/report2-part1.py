import sys, os

if not sys.path[0] == os.path.abspath('..'):
    sys.path.insert(0, os.path.abspath('..'))

from Utils import Utils
from Functions import Functions

# Execution Variables
hitRateList = []
dataBasePercentage = 0.8
totalExec = 20
distributions = ['laplace', 'norm', 'gamma', 'lognorm', 'norm', 'lognorm']

#### SIMULATION CONFIGURATION ####
dataBaseName = "iris"     # iris | column
showNormalPlots = True    # True | False
showPlots300 = True       # True | False

# Column Distributions
data = Utils.getDataBasePandas(dataBaseName)

if(showNormalPlots):
    for i, feature in enumerate(data.columns):
        Utils.plotDistribution(data[feature], feature, distributions[i])

if(dataBaseName == "iris"):
    data300 = Utils.increseDatasetIris(dataBaseName)
else:
    data300 = Utils.increseDatasetColumn(dataBaseName)

if(showPlots300):
    for i, feature in enumerate(data300.columns):
        Utils.plotDistribution(data300[feature], feature, distributions[i])

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

    hitRateList.append(Utils.hitRate(predictResult))
    
# Final Results
print("\nLista das taxas de acerto:")
print(hitRateList)

print("Acurácia: " + str(round(Utils.average(hitRateList), 4)))
print("Desvio Padrão: " + str(round(Utils.standardDeviation(hitRateList), 4)))
