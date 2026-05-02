import numpy as np

def euclidean_distance(x, y):
    """
    Returns: float, the Euclidean distance between x and y.
    """
    x = np.array(x)
    y = np.array(y)

    if len(x) != len(y):
        raise ValueError("...")

    dif = (x-y)
    ed = np.sqrt(np.sum(dif**2))

    return float(ed)