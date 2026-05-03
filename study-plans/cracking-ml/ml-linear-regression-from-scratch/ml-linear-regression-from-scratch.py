import numpy as np

def linear_regression(X, y, lr, epochs):
    """
    Returns: tuple (weights, bias)
    """
    X = np.array(X)
    y = np.array(y)
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0

    for _ in range(epochs):
        y_hat = np.dot(X,w)+b

        dw = (2/n)* X.T @(y_hat - y)
        db = (2/n)*np.sum(y_hat - y)

        w -= lr*dw
        b -= lr*db

    W = np.round(w, 4).tolist()
    B = np.round(float(b),4)
    return (W,B)
    
