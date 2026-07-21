import numpy as np

def bottleneck_block(x, W1, W2, W3, Ws):
    """
    Returns: np.ndarray with bottleneck residual block output (compress, process, expand + skip)
    """
    x = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)
    W3 = np.array(W3)

    if Ws is None:
        shortcut = x
    else:
        Ws = np.array(Ws)
        shortcut = x @ Ws

    y1 = np.maximum(0,x@W1)
    y2 = np.maximum(0,y1@W2)
    y3 = y2@W3

    output = y3+shortcut

    return np.round(np.maximum(0,output),4)
