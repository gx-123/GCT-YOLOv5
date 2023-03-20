import os

path = r"D:\DataSet\helmet\labels\test\\"

# 获取该目录下所有文件，存入列表中
fileList = os.listdir(path)
alldata = []
for i in fileList:
    with open(path + i, "r") as f:  # 打开文件
        data = f.readlines()  # 读取文件
        alldata.append(data)
        for j in data:
            # print(j)
            if j[0] == '0' or j[0] == '1':
                continue
            else:
                print(i)