
import os
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
    current_dir = os.path.dirname(os.path.realpath(__file__))
    nodes = []

    if(onlyGraphic):
        file = open(join(current_dir, 'artificial.data'), 'r')
        
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
        file = open(join(current_dir, 'artificial.data'), 'w')
        for i in range(15):
            x = round(np.random.uniform(0, 0.4), 2)
            y = round(np.random.uniform(0, 0.9), 2)

            file.write(str(x) + "," + str(y) + ",A\n")
            nodes.append([x, y, 'A'])

        for i in range(15):
            x = round(np.random.uniform(0, 0.9), 2)
            y = round(np.random.uniform(0, 0.4), 2)

            file.write(str(x) + "," + str(y) + ",A\n")
            nodes.append([x, y, 'A'])

        for i in range(10):
            x = round(np.random.uniform(0.55, 1), 2)
            y = round(np.random.uniform(0.55, 1), 2)

            file.write(str(x) + "," + str(y) + ",B\n")
            nodes.append([x, y, 'B'])

    file.close()
    return nodes

experiment = generateData(True)

for k in experiment:
    if(k[2] == 'A'):
        color = 'blue'
    else:
        color = 'red'
    plt.scatter(float(k[0]), float(k[1]), c = color)

plt.axis([-0.05, 1.05, -0.05, 1.05])
plt.grid(True)
plt.savefig('Artificial.png')
plt.show()
