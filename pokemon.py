import csv
import matplotlib.pyplot as plt
import function_collect

df = []
with open(r".\pokemon\Pokemon.csv", 'r', encoding="utf8")as f:
    rdr = csv.reader(f)
    for line in rdr:
        df.append(line)
n_row = len(df)
n_col = len(df[0])

myData = []
for i in range(1,n_row):
    row = []
    for j in range(5,n_col):
        val = int(df[i][j])
        row.append(val)
    myData.append(row)
n = len(myData)
c = len(myData[0])
transposedmyData = list(zip(*myData))

total = list(map(int, transposedmyData[0]))
print(total)
'''
hp    = list(map(int, transposedmyData[1]))
atk   = list(map(int, transposedmyData[2]))
dfs   = list(map(int, transposedmyData[3]))
spatk = list(map(int, transposedmyData[4]))
spdfs = list(map(int, transposedmyData[5]))
speed = list(map(int, transposedmyData[6]))
'''

total_sample = function_collect.sample_from_data(total,size = 1000000)

plt.figure(figsize=(20,10))
plt.subplot(2,2,1)
plt.hist(total_sample)
plt.title("sample")
plt.subplot(2,2,2)
plt.hist(total)
plt.title("original")
plt.show()