import numpy as np

def identity_block(x, W1, W2):
    """
    Returns: np.ndarray of shape (batch, channels) with identity residual block output
    """
    x = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)
    
    lp = np.matmul(x,W1.T)
    h = np.maximum(0, lp)
    f = h@W2.T
    res =f+x
    y = np.maximum(0, res)

    return y