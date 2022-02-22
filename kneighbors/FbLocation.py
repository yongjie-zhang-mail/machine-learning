import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 获取数据集
data = pd.read_csv("./data/fb_train.csv")
data.head()
data.describe()
print(data.shape)

# 基本数据处理
# 缩小数据范围
fb_data = data.query("x>2.0 & x<2.5 & y>2.0 & y<2.5")
# 选择时间特征
time = pd.to_datetime(fb_data["time"], unit="s")
time = pd.DatetimeIndex(time)
fb_data["day"] = time.day
fb_data["hour"] = time.hour
fb_data["weekday"] = time.weekday
# 去掉签到较少的地方
place_count = fb_data.groupby("place_id").count()
place_count = place_count[place_count["row_id"] > 3]
fb_data = fb_data[fb_data["place_id"].isin(place_count.index)]
# 确定特征值和目标值
x = fb_data[["x", "y", "accuracy", "day", "hour", "weekday"]]
y = fb_data["place_id"]
# 分割数据集
x_train, x_test, y_train, y_test = train_test_split(x, y)

# 特征工程
# 标准化
# 实例化一个转换器
transfer = StandardScaler()
# 调用 fit_transform 方法
x_train = transfer.fit_transform(x_train)
x_test = transfer.fit_transform(x_test)

# 机器学习 knn + cv
# 实例化一个估计器
estimator = KNeighborsClassifier()
# 调用 GridSearchCV
param_grid = {"n_neighbors": [1, 3, 5, 7, 9]}
GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5)
# 模型训练
estimator.fit(x_train, x_test)

# 模型评估
# 基本评估方式
score = estimator.score(x_test, y_test)
print("预测的准确率：\n", score)
y_pre = estimator.predict(x_test)
print("预测值：\n", y_pre)
print("对比：\n", y_pre == y_test)
