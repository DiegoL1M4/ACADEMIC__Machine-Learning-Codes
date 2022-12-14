from math import floor
from utils.General import General

class KNN:   
    def train(data, percentage):
        # Data division
        total = floor(len(data) * percentage)
        dataTrained = []
        dataToPredict = []

        counter = 1
        for sample in data:
            if(counter <= total):
                dataTrained.append(sample)
            else:
                sample.append('')
                dataToPredict.append(sample)
            counter += 1
        
        return [dataTrained, dataToPredict]
        
    def predict(K_value, dataBase, sample):
        distances = []
        for data in dataBase:
            distance = General.calcDistance(data[0], sample)
            distances.append([distance, data[1]])
        distances = sorted(distances, key=lambda case: case[0])

        # K definition
        totalTypes = []
        result = ''
        for distance in distances:
            if(distances.index(distance) == K_value):
                break

            notFound = True
            for item in totalTypes:
                if(item[0] == distance[1]):
                    item[1] = item[1] + 1
                    notFound = False

            if(notFound):
                totalTypes.append([distance[1], 1, distance[0]]) # name, amount, distance
        totalTypes = sorted(totalTypes, key=lambda case: case[1])

        best = ''
        for item in totalTypes:
            if(best == ''):
                best = item
            else:
                if(item[1] > best[1]):
                    best = item
                elif(item[1] == best[1] and item[2] < best[2]):
                    best = item
                
        result = best[0]

        return result
