
import os
import numpy as np

def className(num):
    if(num == "1\n"):             
        return "psoriasis\n"
    if(num == "2\n"):             
        return "seboreic_dermatitis\n"
    if(num == "3\n"):             
        return "lichen_planus\n"
    if(num == "4\n"):             
        return "pityriasis_rosea\n"
    if(num == "5\n"):             
        return "cronic_dermatitis\n" 
    if(num == "6\n"):             
        return "pityriasis_rubra_pilaris\n"

current_dir = os.path.dirname(os.path.realpath(__file__))
file = open('data/dermatology.data', 'r')
data = []

row = file.readline()
while(row != '' and row != '\n'):
    parts = row.split(",")
    if(parts[33] != '?'):
        data.append(int(parts[33]))
    row = file.readline()

file.close()

mean = int(np.mean(data))
median = int(np.median(data))

# Create new file to replace ? and replace class name
file = open('data/dermatology.data', 'r')
newFile = open('data/dermatologyMod.data', 'w')
row = file.readline()
while(row != '' and row != '\n'):
    parts = row.split(",")

    row = row.replace("?", str(median))
    row = row.replace(parts[34], className(parts[34]))

    newFile.write(row)
    row = file.readline()

file.close()
newFile.close()
