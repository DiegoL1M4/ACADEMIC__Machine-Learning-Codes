
from ntpath import join
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
file = open(join(current_dir, 'breast.data'), 'r')
newFile = open(join(current_dir, 'breastMod.data'), 'w')

# Create new file to remove first column, just id
row = file.readline()
while(row != '' and row != '\n'):
    parts = row.split(",")
    newFile.write(row.replace(parts[0] + ",", ""))
    row = file.readline()

file.close()
newFile.close()
