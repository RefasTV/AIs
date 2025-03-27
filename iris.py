import numpy as np
import random
import matplotlib.pyplot as plt
 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(x, 0)

def relu_deriv(x): # Найти производную relu в точке x
    return (x >= 0).astype(float)

def softmax(x_list):
    x = np.exp(x_list)
    return x / np.sum(x)

def cross_entropy(out, ans):
    return -np.log(out[0, ans])

def to_full(ind, size): # Make a list of zeroes, with 1 in a certain index
    y_full = np.zeros((1, size))
    y_full[0][ind] = 1
    return y_full

from sklearn import datasets
iris = datasets.load_iris()
dataset = [(iris.data[i][None, ...], iris.target[i]) for i in range(len(iris.target))]

INP_DIM = 4
H_DIM = 3
OUT_DIM = 3
 
alpha = 0.001

w1 = np.random.rand(INP_DIM, H_DIM)
b1 = np.random.rand(1, H_DIM)
w2 = np.random.rand(H_DIM, OUT_DIM)
b2 = np.random.rand(1, OUT_DIM)

EPOCHAS = 5000

for i in range(EPOCHAS):
    random.shuffle(dataset)
    for j in range(len(dataset)):
        inp, ans = dataset[j]
        
        # Forward
        t1 = b1 + inp @ w1 
        h1 = relu(t1)

        t2 = b2 + h1 @ w2
        out = softmax(t2) # Find "confidence" of an answer

        E = cross_entropy(out, ans) # Find error

        # Error here means how close AI's answer was to the actual answer

        # Backwards
        ans_full = to_full(ans, OUT_DIM)
        dE_dt2 = out - ans_full
        dE_dw2 = h1.T @ dE_dt2
        dE_db2 = dE_dt2

        dE_dh1 = dE_dt2 @ w2.T
        dE_dt1 = dE_dh1 * relu_deriv(t1)
        dE_dw1 = inp.T @ dE_dt1
        dE_db1 = dE_dt1 


        # Update
        w1 = w1 - alpha * dE_dw1
        b1 = b1 - alpha * dE_db1
        w2 = w2 - alpha * dE_dw2
        b2 = b2 - alpha * dE_db2


def calc_accuracy(dataset, w1, b1, w2, b2):
    random.shuffle(dataset)
    lossarr = []
    right = 0
    for j in range(len(dataset)):
        inp, ans = dataset[j]
        
        # Forward
        t1 = b1 + inp @ w1 
        h1 = relu(t1)

        t2 = b2 + h1 @ w2
        out = softmax(t2) # Find "confidence" of an answer

        if (np.argmax(out) == ans): right += 1

        E = cross_entropy(out, ans) # Find error

        lossarr.append(E)

    return lossarr, right, len(dataset)


lossarr, right, tries = calc_accuracy(dataset, w1, b1, w2, b2)

# Default accuracy is around 98%
print(f"Accuracy: {(right/tries * 100)}%")
plt.plot(lossarr)
plt.show()

