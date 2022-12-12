import sys, os
import random
import pandas as pd

if not sys.path[0] == os.path.abspath('..'):
    sys.path.insert(0, os.path.abspath('..'))

from Functions import Functions
from Utils import Utils

# Variables
data = Utils.getCoordsMatrix("iris")
dataAltered = Utils.copyMatrix(data, 1)

#### SIMULATION CONFIGURATION ####
percentage = 0.1 # 0.1 | 0.3

attributesSize = len(data) * len(data[0])
targetAmount = int(attributesSize * percentage)

# Delete Values
deleted = []
for k in range(targetAmount):
    i = random.randrange(0, len(data))
    j = random.randrange(0, len(data[0]))

    while([i, j] in deleted):
        i = random.randrange(0, len(data))
        j = random.randrange(0, len(data[0]))

    dataAltered[i][j] = "?"
    deleted.append([i, j])

# Valor Esperado
result1 = Functions.CMI(Utils.copyMatrix(dataAltered, 2))

# KNN
result2 = Functions.KNN(Utils.copyMatrix(dataAltered, 1), 30)

# Regressão (Linear Simples)
result3 = Functions.regression(Utils.copyMatrix(dataAltered, 2))

# Result
print("\nCMI:", Utils.RMSE(data, result1))
print("KNN:", Utils.RMSE(data, result2))
print("Regression:", Utils.RMSE(data, result3))

print("")
