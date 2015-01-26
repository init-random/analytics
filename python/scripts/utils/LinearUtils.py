import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
import numpy.random as random
import pandas as pd


def scatterplot(x, y):
    plt.scatter(x, y)


def decay(x):
    val = .9 * 2**-abs(x)
    return max(val, 0.00005)


def linear_regression(x, y):
    model = linear_model.LinearRegression()
    fit = model.fit(x, y)
    return np.array([fit.intercept_, fit.coef_]).flatten()


def linear_regression_matrix_solve(x, y):
    # beta_0 is the intercept where x_0 = 1
    xm = np.column_stack((np.ones([x.shape[0]]), x))
    ym = y
    xm = np.matrix(xm)
    return (xm.transpose().dot(xm)).getI().dot(xm.transpose()).dot(ym)


def linear_regression_stoc_grad_desc(x, y):
    xm = x
    ym = y
    if type(x) == pd.core.frame.DataFrame:
        xm = np.matrix(xm.values)
        ym = np.matrix(y.values)
    xm = np.column_stack((np.ones([x.shape[0]]), x))

    dims = xm.shape
    dims_len = len(dims)
    features = (dims[1] if dims_len == 2 else 1)
    # init some small value
    theta = np.array(0.2 * random.random(features))
    # learning rate
    alpha = 0.0000006
    count = 0
    try:
        while True:
            # pre-pend 1, which is bias or intercept
            # stochastic gradient descent
            for idx in range(dims[0]):
                count += 1
                error = h(xm[idx, :], theta) - ym[idx, 0]
                delta = xm[idx, :] * error
                # _theta = theta.copy()
                theta[:] = theta - alpha * delta

                if count > 50000: # stopping condition
                    raise Exception
                if idx % 100 == 0:
                    print('d ' + str(np.dot(theta - theta, theta - theta)))
                    print(str(theta))
                    print(str(theta))

    except Exception as e:
        print(e)

    print('T: '+ str(theta))
    slope = theta[1]
    intercept = theta[0]
    return (slope, intercept)


def linear_regression_batch_grad_desc(x, y):
    slope = 0
    intercept = 0
    x['b0'] = 1
    cl = x.columns.tolist()
    b0 = cl.pop()
    cl.insert(0, b0)
    xm = x[cl] # fix column order
    xm = np.matrix(xm.values)
    ym = np.matrix(y.values)

    dims = xm.shape
    dims_len = len(dims)
    features = (dims[1] if dims_len == 2 else 1)
    # init some small value
    theta = np.matrix(0.2 * random.random(features))
    # theta = np.matrix([108., -2.])
    # learning rate
    alpha = 0.01
    count = 0
    try:
        while True:
            print(str(count))
            # pre-pend 1, which is bias or intercept
            # stochastic gradient descent
            update = 0
            for idx in range(dims[0]):
                count += 1
                d = ym[idx] - h(theta, xm[idx])
                update += d * xm[idx]
                if idx % 100 == 0:
                    print(str(theta))
            tmp_theta = theta + alpha * update
            diffm = (tmp_theta - theta)
            diff = diffm * diffm.getT()
            theta = tmp_theta
            if count > 50000: # stopping condition
                raise Exception

    except:
        pass

    print('T: '+ str(theta))
    slope = theta[0, 1]
    intercept = theta[0, 0]
    return (slope, intercept)


def h(theta, x):
    # hypothesis function
    return np.dot(theta, x)


def xxx():
    x = np.array([[1., 200.],
                  [1., 300.],
                  [1., 400.]])
    m = np.mean(x[:, 1])
    sd = np.std(x[:, 1])
    print(str(m))
    print(str(sd))
    x[:, 1] = (x[:, 1] - m)/sd
    y = np.array([[5.08],
                  [6.9],
                  [9.1]])
    b = np.array([2.1, -2.1])

    eps = 0.03 # step size
    precision = 0.00001
    c = 0
    while c < 1500: # abs(x_new - x_old) > precision:
        idx = c % 3
        # b_old = b_new
        error = np.dot(x[idx, :], b) - y[idx, 0]
        delta = x[idx, :] * error
        b[:] = b - eps * delta
        if c % 100 == 0:
            print(str(b))
        c += 1
    print(str(b))



