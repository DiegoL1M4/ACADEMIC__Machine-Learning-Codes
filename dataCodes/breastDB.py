
file = open('data/breast.data', 'r')
newFile = open('data/breastMod.data', 'w')

# Create new file to remove first column, just id
row = file.readline()
while(row != '' and row != '\n'):
    parts = row.split(",")
    newFile.write(row.replace(parts[0] + ",", ""))
    row = file.readline()

file.close()
newFile.close()
