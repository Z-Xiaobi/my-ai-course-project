# -*- coding='utf-8' -*-
# @Time    : 11/27/21 17:03
# @Author  : Xiaobi Zhang
# @FileName: perceptron_logistic_regression.py

import numpy as np
import re

## Get input
inputs = input()
inputs = [p for p in re.split(r'\)| \(', inputs) if p.strip()]
# print(inputs)

data = []
for d in inputs[1:]:
    d = d.replace(" ", "").split(',')
    d = [int(num) for num in d]
    data.append(d)

data = np.array(data)


def perceptron(X, y):
    """
    update w for each predication on one data point
    :param X: features, dx2 matrix
    :param y: labels for given X
    :return:
    """
    n = X.shape[0]
    w = np.zeros((2,))

    def pred(x, curr_w):
        # print("x:",x," w:", curr_w)
        activation = x[0] * curr_w[0] + x[1] * curr_w[1]
        # print("activation:",activation)
        if activation >= 0:
            return 1
        else:
            return -1

    for _ in range(n * 100):
        count = n # number of correct prediction
        for i in range(n):
            prediction = pred(X[i], w)
            # print("true label: ", y[i])
            # print("predication: ", prediction)
            if prediction != y[i]:
                w[0] = w[0] + y[i] * X[i][0]
                w[1] = w[1] + y[i] * X[i][1]
                count -= 1
        if count == n:
            break
    return w


def logistic(X, y):
    n = X.shape[0]
    w = np.zeros((2,))
    lr = 0.1

    # prediction
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    # update
    for _ in range(100):
        for i in range(n):
            # print("z: ", w[0] * X[i][0] + w[1] * X[i][1])
            prediction = sigmoid(w[0] * X[i][0] + w[1] * X[i][1])
            # print(prediction)
            # 1 -1 ---> 1 0
            label = 0 if (y[i] == -1) else y[i]
            # label = y[i]
            # if int(prediction - label) == 0: break
            w[0] = w[0] - lr * X[i][0] * (prediction - label)
            w[1] = w[1] - lr * X[i][1] * (prediction - label)

    # prediction on updated weights
    result = []
    for i in range(n):
        result.append(sigmoid(X[i].dot(w)))
    return result


if inputs[0] == 'P':
    weights = perceptron(data[:, :2], data[:, 2])
    print(", ".join(str(weight) for weight in weights))
elif inputs[0] == 'L':
    prob = logistic(data[:, :2], data[:, 2])
    print(" ".join(str(round(p,2)) for p in prob))
else:
    raise ValueError("wrong operation")

## tests
# print(int(" 2")) # 2
# print(int("-1")) # -1
