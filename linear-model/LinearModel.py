from sklearn.linear_model import LinearRegression

# 获取数据
x = [[80, 86],
     [82, 80],
     [85, 78],
     [90, 90],
     [86, 82],
     [82, 90],
     [78, 80],
     [92, 94]]
y = [84.2, 80.6, 80.1, 90, 83.2, 87.6, 79.4, 93.4]

# 模型训练
# 实例化一个估计器
estimator = LinearRegression()
# 调用 fit 方法，进行模型训练
estimator.fit(x, y)

# 查看系数值
print("系数是：\n", estimator.coef_)

# 预测
print("预测值是：\n", estimator.predict([[80, 100]]))
