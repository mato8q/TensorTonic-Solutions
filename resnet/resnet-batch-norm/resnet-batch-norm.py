import numpy as np

def batch_norm_block(x, W1, W2, gamma1, beta1, gamma2, beta2, mode):
    """
    Returns: np.ndarray of same shape as input with batch-normalized and skip-connected output
    """
    x = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)
    gamma1 = np.array(gamma1)
    gamma2 = np.array(gamma2)
    beta1 = np.array(beta1)
    beta2 = np.array(beta2)

    def bn(x, gamma,beta):
        mean = x.mean(axis=0)
        var = x.var(axis = 0)
        x_hat = (x - mean)/np.sqrt(var+ 1e-5)

        return gamma*x_hat + beta
        
    if mode == 'post':
        # x -> conv -> BN -> ReLU 
        #-> conv -> BN -> 
        # Add -> ReLU -> y
        h = np.maximum(0, bn(x@W1, gamma1, beta1))
        z = bn(h@W2, gamma2, beta2)
        y = np.maximum(0, z+x)
        return { 'output' : np.round(y,4).tolist(), 'mode':'post'}

    elif mode == 'pre':
        # x -> BN -> ReLU -> conv
        # BN -> ReLU
        # conv -> add -> y

        #BN(x) -> ReLU -> conv (@W1)
        h = np.maximum(0,bn(x,gamma1,beta1))@W1

        #BN(h) -> ReLU -> conv(@W2)
        z =  np.maximum(0, bn(h, gamma2, beta2))@W2

        #add -> y
        y = z+x
        return { 'output':np.round(y,4).tolist(), 'mode':'pre'}
    