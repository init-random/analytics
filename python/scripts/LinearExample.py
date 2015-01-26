import scripts.data.DataLoad as datasets
import scripts.utils.LinearUtils as lu
import sys


def main():
    (x, y) = datasets.housing_data()
    x = x[['living_area']]

    lu.scatterplot(x, y)

    print('sklearn linear regression model')
    lu.linear_regression(x, y)

    print('matrix solution linear regression model')
    lu.linear_regression_matrix_solve(x, y)

    print('stochastic gradient descent linear regression model')
    lu.linear_regression_stoc_grad_desc(x, y)

    print('gradient descent linear regression model')
    lu.linear_regression_batch_grad_desc(x, y)


if __name__ == '__main__':
    sys.exit(main())
