import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute the softmax of a numpy array.

    Args:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Softmax of the input array.
    """
    # Subtract max for numerical stability
    # e_x = np.exp(x - np.max(x))      
    # return e_x / e_x.sum(axis=-1, keepdims=True)

    # 防溢出处理
    x_stable = x - np.max(x)
    exp_x = np.exp(x_stable)
    sum_exp_x = np.sum(exp_x)
    result = exp_x / sum_exp_x
    return result


if __name__ == "__main__": 
    # print("hello")
    x = [2.0, 1.0, 0.1]
    y = softmax(x)
    print(y)


