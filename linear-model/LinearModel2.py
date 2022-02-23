"""
获取数据
数据基本处理（数据集划分）
特征工程（标准化）
机器学习（线性回归）
模型评估
"""

import joblib
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def linear_model1():
    """
    正规方程
    :return: None
    """
    x_test, x_train, y_test, y_train = prepare()

    # 机器学习（线性回归）
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)
    print("这个模型的偏置是：\n", estimator.intercept_)

    assess(estimator, x_test, y_test)


def linear_model2():
    """
    梯度下降
    :return: None
    """
    x_test, x_train, y_test, y_train = prepare()

    # 机器学习
    estimator = SGDRegressor(max_iter=1000)
    estimator.fit(x_train, y_train)
    print("这个模型的偏置是：\n", estimator.intercept_)

    assess(estimator, x_test, y_test)


def linear_model3():
    """
    岭回归
    :return: None
    """
    x_test, x_train, y_test, y_train = prepare()

    # 机器学习
    # estimator = Ridge()
    estimator = RidgeCV(alphas=(0.001, 0.1, 1, 10, 100))
    estimator.fit(x_train, y_train)
    print("这个模型的偏置是：\n", estimator.intercept_)

    assess(estimator, x_test, y_test)


def model_dump_load():
    """
    模型保存和加载
    :return: None
    """
    x_test, x_train, y_test, y_train = prepare()

    # 机器学习
    estimator = Ridge()
    # estimator = RidgeCV(alphas=(0.001, 0.1, 1, 10, 100))
    estimator.fit(x_train, y_train)
    print("这个模型的偏置是：\n", estimator.intercept_)

    # 模型保存
    joblib.dump(estimator, "./model/ridge_model.pkl")
    # 模型加载
    estimator = joblib.load("./model/ridge_model.pkl")

    assess(estimator, x_test, y_test)


def prepare():
    """准备数据，数据集划分，特征工程
    """
    # 获取数据
    boston = load_boston()
    # 数据基本处理（数据集划分）
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2)
    # 特征工程（标准化）
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)
    return x_test, x_train, y_test, y_train


def assess(estimator, x_test, y_test):
    """模型评估
    """
    # 模型评估
    y_pred = estimator.predict(x_test)
    print("预测值：\n", y_pred)
    score = estimator.score(x_test, y_test)
    print("准确率是：\n", score)
    mse = mean_squared_error(y_test, y_pred)
    print("均方误差是：\n", mse)


if __name__ == '__main__':
    # linear_model1()
    # linear_model2()
    # linear_model3()
    model_dump_load()
