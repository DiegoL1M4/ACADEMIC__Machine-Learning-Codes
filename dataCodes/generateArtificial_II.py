
import numpy as np
import matplotlib.pyplot as plt

from ntpath import join

def isNumber(value):
    try:
        float(value)
    except ValueError:
        return False
    return True

def generateData(onlyGraphic):
    nodes = []

    if(onlyGraphic):
        file = open('data/artificialII.data', 'r')
        
        row = file.readline()
        while(row != '' and row != '\n'):
            sample = []
            elements = row.replace('\n', '').split(',')
            
            for e in elements:
                if(isNumber(e)):
                    sample.append(float(e)) #coord
                else:
                    sample.append(e) #type

            nodes.append(sample)
            row = file.readline()
    
    if(not onlyGraphic):
        file = open('data/artificialII.data', 'w')
        for i in range(15):
            x = round(np.random.uniform(0.1, 0.3), 2)
            y = round(np.random.uniform(0.55, 0.8), 2)

            file.write(str(x) + "," + str(y) + ",A\n")
            nodes.append([x, y, 'A'])

        for i in range(15):
            x = round(np.random.uniform(0.64, 0.8), 2)
            y = round(np.random.uniform(0.55, 0.8), 2)

            file.write(str(x) + "," + str(y) + ",B\n")
            nodes.append([x, y, 'B'])

        for i in range(15):
            x = round(np.random.uniform(0.35, 0.5), 2)
            y = round(np.random.uniform(0.15, 0.4), 2)

            file.write(str(x) + "," + str(y) + ",C\n")
            nodes.append([x, y, 'C'])

    file.close()
    return nodes



##### MAIN #####

experiment = generateData(True) # OnlyGraphic: True | False

for k in experiment:
    if(k[2] == 'A'):
        color = 'blue'
    if(k[2] == 'B'):
        color = 'red'
    if(k[2] == 'C'):
        color = 'black'
    plt.scatter(float(k[0]), float(k[1]), c = color)

plt.axis([-0.05, 1.05, -0.05, 1.05])
plt.grid(True)
plt.savefig('graphics/Artificial2.png')
plt.show()
