import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """ 
    Apply layer normalization.
    """
    mu = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    norm = (x-mu)/(np.sqrt(var+eps))

    return gamma*norm+beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    Q_proj = np.dot(Q,W_q)
    K_proj = np.dot(K,W_k)
    V_proj = np.dot(V,W_v)

    batch, seq, d_model = Q.shape
    d_k = d_model//num_heads

    Q_proj = Q_proj.reshape(batch, seq, num_heads, d_k).transpose(0,2,1,3)
    K_proj = K_proj.reshape(batch, seq, num_heads, d_k).transpose(0,2,1,3)
    V_proj = V_proj.reshape(batch, seq, num_heads, d_k).transpose(0,2,1,3)

    score = np.matmul(Q_proj, K_proj.transpose(0,1,3,2))/np.sqrt(d_k)
    
    weight = softmax(score, axis=-1)
    head = np.matmul(weight, V_proj)
    result = head.transpose(0,2,1,3).reshape(batch, seq, d_model)
    return np.dot(result,W_o)

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    h = np.dot(x,W1)+b1
    a = np.maximum(0,h)
    ffn = np.dot(a,W2)+b2

    return ffn
    

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    att_output = multi_head_attention(x,x,x, W_q, W_k,W_v,W_o,num_heads)
    residual = x+att_output
    x_prime = layer_norm(residual, gamma1, beta1)

    ffn_output = feed_forward(x_prime, W1, b1, W2, b2)
    output = layer_norm(x_prime+ ffn_output, gamma2, beta2)
    return output