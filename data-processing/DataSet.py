import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris, fetch_20newsgroups
from sklearn.model_selection import train_test_split

# 小数据集获取
iris = load_iris()
# print(iris)

# 大数据集获取
news = fetch_20newsgroups()
# print(news)

# 数据集属性
# # 数据集描述
# print(iris.DESCR)
# # 特征值名称&特征值
# print(iris.feature_names)
# print(iris.data)
# # 目标值名称&目标值
# print(iris.target_names)
# print(iris.target)

# 数据可视化
# 数据类型转换，把数据用DataFrame存储
iris_data = pd.DataFrame(data=iris.data,
                         columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
iris_data["target"] = iris.target


# print(iris_data)


def iris_plot(data, col1, col2):
    sns.lmplot(x=col1, y=col2, data=data, hue="target", fit_reg=False)
    plt.title("鸢尾花数据展示")
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.show()


# iris_plot(iris_data, "sepal length (cm)", "petal width (cm)")
# iris_plot(iris_data, 'sepal width (cm)', 'petal length (cm)')

# 数据集划分
# test_size: 测试集百分比;
# random_state: 随机数种子, 若一样则结果集固定一样;
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=2)
# print("训练集特征值：\n", x_train)
# print("测试集特征值：\n", x_test)
# print("训练集目标值：\n", y_train)
# print("测试集目标值：\n", y_test)

print("训练集目标值的形状：\n", y_train.shape)
print("测试集目标值的形状：\n", y_test.shape)
