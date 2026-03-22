import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    n = len(y)
    w = np.zeros(X.shape[1])
    b = 0
    for _ in range(steps):
        z = X@w+b  # ใช้ @ แทน * เพราะต้องคูณแบบ metrix
        p = _sigmoid(z)
    
        #gradient descent
        dw = (1/n)*((X.T)@(p-y))
        db = (1/n)*(np.sum(p-y))

        w -= lr*dw
        b -= lr*db
    
    return (w,b)