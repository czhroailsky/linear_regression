import numpy as np
import matplotlib.pyplot as plt

class LinearModel(object):

    """ eta <- learning rate: is an hyperparameter that controls how much we are
    adjusting the weights with respect the loss gradient.
        n_iter <- Number of iteration to adjust the weights (beta)"""

    def __init__(self, eta=0.1, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    """
    Approximate a target value $Y$ as a linear function of X:
    $$\hat Y = \hat \beta_0 + \sum_{j=1}^{p} X_j \hat \beta_j$$
    where $p$ is the number of features, and the term $\hat \beta$
    is then intercept, also known as the bias. For convenience we
    include the constant variable 1 in $X$ which is a vector of inputs
    $X^T = \left( X_1, X_2, \ldots, X_p \right)$, include $\hat \beta_0$
    in the vector of coefficients $\hat \beta$, and then write the linear
    model in vector form as an inner product $$\hat Y = X^T \hat \beta $$
    """

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1) ## Insert the constant variable 1
        B = np.ones(X.shape[1]) ## Initialize vector beta

        print('B: %s' %(B))
        print('RSS(B): %s' %(self.rss(X, y, B)))

        B = self.unq_solution(X, y)
        print('B_update: %s' %(B))
        print('RSS(B_update): %s' %(self.rss(X, y, B)))

        return(B)

    """
    Least squares to fit the liear model to a set of training data. In this
    approach, we pick the coefficients $\beta $ to minimize the residual sum of
    squares
    $$ RSS \left( \beta \right) = \sum_{i=1}^{N} \left( y_i - x_i^T \beta \right) ^ 2 $$

    $RSS\left( \beta \right)$ is a quadratic function of the parameters, and hence
    its minimum always exists, but may not be unique.
    """
    def rss(self, X, y, B):
        RSS = (y - X.dot(B)).dot(np.transpose(y - X.dot(B)))
        return(RSS)

    def unq_solution(self, X, y):
        XT_X = np.transpose(X).dot(X)
        XT_X_det = np.linalg.det(XT_X) ## If point product between X transpose and X is nonsingular, the inverse of the resulting matrix can be computed

        if XT_X_det != 0: ## Verify if matrix is nonsingular
            XT_X_inv = np.linalg.inv(XT_X)
            XT_y = np.transpose(X).dot(y)
            B = XT_X_inv.dot(XT_y)
        else:
            print('Matrix is singular.')

        return(B)

if __name__ == "__main__":

    X = np.array([[73],
                 [91],
                 [87],
                 [102],
                 [69]], dtype = 'float32') ## N x p matrix, N <- instances, p <- attributes

    y = np.array([0, 0, 0, 1, 1], dtype = 'float32') ## N-vector of the outputs in the training set

    print('X: %s' %(X))
    print('y: %s' %(y))

    print('\nComputing coefficients beta: ')
    B = LinearModel().fit(X, y)
    print('\nB: %s' %(B))

    ## Predictions
    print('\nPredicted outputs: ')
    y_hat = B[0] + B[1] * X
    print('y_hat: %s' %(y_hat))

    plt.scatter(X, y,  color='black')
    plt.plot(X, y_hat, color='blue', linewidth=3)
    plt.show()
