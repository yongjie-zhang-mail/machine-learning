import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

data = pd.read_csv(filepath_or_buffer="./data/datingTestSet2.txt", delimiter='\t')
print(data)

# 归一化
# 实例化一个转换器
minMaxTransfer = MinMaxScaler(feature_range=(0, 1))
# 调用 fit_transform 方法
minMaxData = minMaxTransfer.fit_transform(data[["milage", "Liters", "Consumtime"]])
# print("归一化之后的数据：\n", minMaxData)

# 标准化
# 实例化一个转换器
standardTransfer = StandardScaler()
# 调用 fit_transform 方法
standardData = standardTransfer.fit_transform(data[["milage", "Liters", "Consumtime"]])
print("标准化之后的数据：\n", standardData)
