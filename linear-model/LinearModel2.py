"""
获取数据
数据基本处理（数据集划分）
特征工程（标准化）
机器学习（线性回归）
模型评估
"""

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def linear_model1():
    """
    正规方程
    :return: None
    """
    # 获取数据
    boston = load_boston()
    # 数据基本处理（数据集划分）
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2)
    # 特征工程（标准化）
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)
    # 机器学习（线性回归）
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)
    # 模型评估
    y_pred = estimator.predict(x_test)
    print("预测值：\n", y_pred)
    score = estimator.score(x_test, y_test)
    print("准确率是：\n", score)
    mse = mean_squared_error(y_test, y_pred)
    print("均方误差是：\n", mse)


if __name__ == '__main__':
    linear_model1()
