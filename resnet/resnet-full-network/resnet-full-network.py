import numpy as np

def resnet_forward(x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc):
    """
    Returns: np.ndarray of shape (batch, num_classes) with classification logits
    """
    x = np.array(x)
    conv1 = np.array(conv1)
    W1_b1 = np.array(W1_b1)
    W2_b1 = np.array(W2_b1)
    W1_b2 = np.array(W1_b2)
    W2_b2 = np.array(W2_b2)
    Ws_b2 = np.array(Ws_b2)
    fc = np.array(fc)

    """
    stage ในชีวิตจริง คือการจัดกลุ่ม blocks
    stage 1 = [block1, block2]
    stage 2 = [block3, block4]
    
    โจทย์มีแค่ 2 blocks
    W1_b1, W2_b1 -> block1
    W1_b2, W2_b2 -> block2
    """ 
    #1 conv
    h = np.maximum(0, x@conv1)

    #2 block1 identity
    h = np.maximum(0, np.maximum(0,h@W1_b1)@W2_b1 +h) 
    #3 block2 projection
    s = h@Ws_b2
    h = np.maximum(0, np.maximum(0, h@W1_b2)@W2_b2+s)

    #4 fc
    output = h@fc

    return np.round(output,4)
    
    
