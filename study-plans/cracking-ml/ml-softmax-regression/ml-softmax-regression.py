import numpy as np

def softmax_regression(X, y, n_classes, lr=0.01, n_iters=1000):
    """
    Returns: tuple (weights, bias) where weights is a 2D list (d x K) and bias is a list of length K
    """
    X = np.array(X)
    y = np.array(y)
    n = X.shape[0]
    d = X.shape[1]
    K = n_classes
    W = np.zeros((d,K))
    b = np.zeros(K)
    Y = np.zeros((n,K))
    Y[np.arange(n),y] = 1 #แปลงเป็นone-hot

    for _ in range(n_iters):
        Z = X@W + b
        #find row-wise max
        m = np.max(Z, axis=1, keepdims = True)
        #deleted max from z
        Z_shifted = Z-m
        #find expo
        P_exp = np.exp(Z_shifted)
        sum_exp = np.sum(P_exp, axis = 1,keepdims=True)
        P = P_exp/sum_exp
        #compute gradient
        diff = P-Y
        dW = (1/n)*(X.T@diff)
        db = (1/n)*np.sum(diff, axis=0)
        W-=lr*dW
        b-=lr*db

    return W.tolist(), b.tolist()
