import numpy as np

def cosine_similarity(a, b):
    """
    Returns: float in [-1, 1], cosine similarity between a and b.
    """
    a = np.array(a)
    b = np.array(b)

    norm_a = np.sqrt(np.sum(a**2))
    norm_b = np.sqrt(np.sum(b**2))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    cos = (np.dot(a,b))/(norm_a*norm_b)

    return float(cos)