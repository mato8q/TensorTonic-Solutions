import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    S = torch.matmul(Q,K.transpose(1,2))
    d_k = math.sqrt(K.shape[-1])
    score = S / d_k
    W = F.softmax(score, dim =-1)
    O = torch.matmul(W,V)
    
    return O