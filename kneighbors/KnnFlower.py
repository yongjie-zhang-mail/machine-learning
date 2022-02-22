"""
获取数据集
数据基本处理
特征工程
机器学习（模型训练）
模型评估
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 获取数据集
iris = load_iris()

# 数据基本处理
# 数据分割
# x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22, test_size=0.2)
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

# 特征工程
# 实例化一个转换器
transfer = StandardScaler()
# 调用 fit_transform 方法
x_train = transfer.fit_transform(x_train)
x_test = transfer.fit_transform(x_test)

# 机器学习（模型训练）
# 实例化一个估计器
estimator = KNeighborsClassifier()

# 调用交叉验证网格搜索模型
param_grid = {"n_neighbors": [1, 3, 5, 7, 9]}
estimator = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=10, n_jobs=-1)

# 模型训练
estimator.fit(x_train, y_train)

# 模型评估
# 输出预测值
y_pre = estimator.predict(x_test)
print("预测值是：\n", y_pre)
print("真实值是：\n", y_test)
print("预测值和真实值对比：\n", y_pre == y_test)
# 输出准确率
ret = estimator.score(x_test, y_test)
print("准确率是：\n", ret)

# 其它评价指标
print("最好的模型：\n", estimator.best_estimator_)
print("最好的结果：\n", estimator.best_score_)
# print("整体模型结果：\n", estimator.cv_results_)
