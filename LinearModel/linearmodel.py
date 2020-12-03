import numpy as np


class LinearRegression:
    """
    Linear Regression Using Normal Equation
    """

    def __init__(self, fit_intercept=False):
        self.weights = None
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """
        Fit the regression weights via maximum likelihood
        and Normal Equation

        :param X: {array-like}, shape = [n_examples, n_features]
                Training Vectors, where n_examples is the number of examples and n_features is the number of features
        :param y: Target Value
        :return: weights and intercept
        """
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        normal_equation = np.linalg.inv(X.T.dot(X)).dot(X.T)
        self.weights = np.dot(normal_equation, y)

    def predict(self, X):
        """
        Using the trained model to generate predictions on a new collection of data points

        :param X: {array-like}, shape = [n_examples, n_features]
                Training Vectors, where n_examples is the number of examples and n_features is the number of features
        :return:
        """
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        return np.dot(X, self.weights)


class RidgeRegression:
    """
    A ridge regression model fit by Normal Equation
    """
    def __init__(self, alpha=1, fit_intercept=False):
        self.weights = None
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """
        Fit the regression coefficients via maximum likelihood

        :param X: {array-like} 'ndarray <numpy.ndarray> of shape: [n_examples, n_features]
        :param y: {array-like} 'ndarray <numpy.ndarray> of shape: [n_examples, ]
        :return: self object
        """
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        A = self.alpha * np.eye(X.shape[1])
        inverse = np.linalg.inv(X.T.dot(X) + A).dot(X.T)
        self.weights = inverse.dot(y)

    def predict(self, X):
        """
        Use the trained model to generate predictions on a new collection of data points

        :param X: {array-like} 'ndarray <numpy.ndarray>' of shape: [n_examples, n_features]
        :return: self
        """
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        return np.dot(X, self.weights)


class LogisticRegression:
    """
    Logistic Regression Classifier Using Gradient Descent.

    Parameters:
    -----------------
    eta: float
        Learning rate (between 0.0 and 1.0)

    n_iter: int
        Passes over the training dataset.

    random_state: int
        Random number generator seed for random weight
        initialization

    Attributes
    -----------------
    w_ : 1d-array
        Weights after fitting

    cost_ : list
        Logistic cost function value in each epoch.
    """

    def __init__(self, eta=0.5, n_iter=100, random_state=42):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit Training data

        :param X: {array-like}, shape = [n_examples, n_features]
                Training Vectors, where n_examples is the number of examples and n_features is the number of features
        :param y: {array-like}, shape = [n_examples]
                Target values.
        :return: self: object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for _ in range(self.n_iter):
            net_input = self._net_input(X)
            output = self._activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()

            # we compute the logistic 'cost' now
            # instead of the sum of squared errors cost
            cost = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))))
            self.cost_.append(cost)

        return self

    def _net_input(self, X):
        """
        Calculate the net input
        :param X: {array-like} shape = [n_examples, n_features]
                Training Vectors, where n_examples is the number of examples and n_features is the number of features
        :return: {array-like} net input matrix
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def _activation(self, z):
        """
        Compute logistic sigmoid activation
        :param z:
        :return:
        """
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """
        Return class label after unit step
        :param X: {array-like} shape = [n_examples, n_features]
                Training Vectors, where n_examples is the number of examples and n_features is the number of features
        :return: label class
        """
        return np.where(self._activation(self._net_input(X)) >= 0.5, 1, 0)


if __name__ == '__main__':
    # test
    pass
