import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
import numpy.random as random
import pandas as pd


def scatterplot(x, y):
    plt.scatter(x, y)


def decay(x):
    """define a decay function for the learning rate (alpha)"""
    return -1


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
    xm = None
    ym = None
    if type(x) == pd.core.frame.DataFrame:
        xm = np.matrix(x.values)
        ym = np.matrix(y.values)
        xm = np.column_stack((np.ones([xm.shape[0]]), xm))
    else:
        xm = np.column_stack((np.ones([x.shape[0]]), x))
        ym = y

    m = np.mean(xm, 0)
    sd = np.std(xm, 0)
    m[0, 0] = 0
    sd[0, 0] = 1
    xm = (xm - m)/sd

    dims = xm.shape
    dims_len = len(dims)
    features = (dims[1] if dims_len == 2 else 1)
    # init some small value
    theta = np.array(0.2 * random.random(features))
    # learning rate
    alpha = 0.001
    precision = 0.0001
    count = 0
    try:
        prev_mse = 0
        curr_mse = 0
        while True:
            # pre-pend 1, which is bias or intercept
            # stochastic gradient descent
            for idx in range(dims[0]):
                curr_mse = 0
                count += 1
                error = h(xm[idx, :], theta) - ym[idx, 0]
                curr_mse += error[0,0] ** 2
                delta = np.array(xm[idx, :]).flatten() * error[0, 0]
                # _theta = theta.copy()
                theta[:] = theta - alpha * delta

                if count > 500000: # stopping condition
                    print('stop iterations')
                    raise Exception
            if abs(prev_mse - curr_mse) < precision: # stopping condition
                raise Exception
            prev_mse = curr_mse

    except Exception as e:
        print(e)

    sd = np.array(sd).flatten()
    m = np.array(m).flatten()
    xhat = np.ones([1, xm.shape[1]])
    xnew = xhat * sd + m + .00001
    xnew = np.ones([2, 2])
    for i in range(2):
        xnew[i, 1:] = m[1:] + (sd[1:]/4) * i
    xh = (xnew - m)/sd
    thetanew = np.linalg.inv(xnew.dot(xnew)).dot(xnew).dot(xh).dot(theta)
    slope = thetanew[1]
    intercept = thetanew[0]

    return intercept, slope


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
            if count > 500000: # stopping condition
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





