import math
import torch

def densenet_channel_counts(stem_channels: int, growth_rate: int,
                            block_layers: list, compression: float) -> torch.Tensor:
    """
    Returns the int64 channel count at every DenseNet stage.
    """
    # dense capacity ที่ไหลผ่าน densenet & inside concatenation along the channel axis
    # store every val
    
    C = stem_channels 
    result = [stem_channels]
    for i,n in enumerate(block_layers):  
        C+= (n * growth_rate )
        result.append(C)
        if i< len(block_layers)-1:
            C = math.floor(compression*C)
            result.append(C)
    return torch.tensor(result, dtype= torch.int64)
            

    # compression @ transition
    # no need to compute conv
    

    
    