import numpy as np

def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITH skip connections.
    Gradient at layer l = sum of paths through network
    """
    result = x.copy().astype(float)

    for grad in gradients_F:
        result = result+ np.dot(result, grad)
    return result
    # gradient ต้องผ่าน Jacobian ทุกครั้ง
    # ถ้า Jacobian เล็ก → หดเล็กลงเรื่อยๆ

def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITHOUT skip connections.
    """
    result = x.copy().astype(float)

    for grad in gradients_F:
        result = np.dot(result, grad)
    return result

    # gradient มีทางลัด ไม่ต้องผ่าน Jacobian
    # ค่าไม่หายไปแม้ Jacobian จะเล็กแค่ไหน