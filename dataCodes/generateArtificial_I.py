
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
        file = open('data/artificial1.data', 'r')
        
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
        file = open('data/artificial1.data', 'w')
        for i in range(6):
            x = round(np.random.uniform(0, 0.25), 2)
            y = round(np.random.uniform(0.6, 0.9), 2)

            file.write(str(x) + "," + str(y) + ",A\n")
            nodes.append([x, y, 'A'])

        for i in range(8):
            x = round(np.random.uniform(0.01, 0.3), 2)
            y = round(np.random.uniform(0.01, 0.3), 2)

            file.write(str(x) + "," + str(y) + ",A\n")
            nodes.append([x, y, 'A'])

        for i in range(6):
            x = round(np.random.uniform(0.6, 1), 2)
            y = round(np.random.uniform(0, 0.25), 2)

            file.write(str(x) + "," + str(y) + ",A\n")
            nodes.append([x, y, 'A'])

        for i in range(10):
            x = round(np.random.uniform(0.64, 0.9), 2)
            y = round(np.random.uniform(0.64, 0.9), 2)

            file.write(str(x) + "," + str(y) + ",B\n")
            nodes.append([x, y, 'B'])

    file.close()
    return nodes



##### MAIN #####

experiment = generateData(True) # OnlyGraphic: True | False

for k in experiment:
    if(k[2] == 'A'):
        color = 'blue'
    else:
        color = 'red'
    plt.scatter(float(k[0]), float(k[1]), c = color)

plt.axis([-0.05, 1.05, -0.05, 1.05])
plt.grid(True)
plt.savefig('graphics/Artificial1.png')
plt.show()
