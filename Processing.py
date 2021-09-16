import pandas as pd
import numpy as np

data = pd.read_excel('./大装置除氧床溶解氧数据集.xlsx', header=0)

data = np.array(data)
rowNum = len(data)
colNum = len(data[0])

output = np.empty(shape=(rowNum*4, colNum-2))

for index, row in enumerate(data):
    for i in range(4):
        if i == 0:
            height = 100
        elif i == 1:
            height = 70
        elif i == 2:
            height = 40
        else:
            height = 0

        curRow = index*4 + i
        for j in range(5):
            output[curRow][j] = row[j]
        output[curRow][5] = height
        output[curRow][6] = row[i+5]
result = pd.DataFrame(output)
result.to_excel('./大装置处理后数据集.xlsx')


