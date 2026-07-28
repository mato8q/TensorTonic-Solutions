def lasso_regression(X, y, lr, epochs, alpha):
    """
    Perform Lasso Regression using gradient descent with L1 subgradient.
    Returns: tuple of (weights_list, bias_float)
    """
    n = len(X)
    d = len(X[0])
    w = [0.0] * d
    b = 0.0
    for _ in range(epochs):
        # 1-2. y_hat และ error
        error = []
        for i in range(n):
            y_hat = b                      # ← จุดสำคัญ: เริ่มสะสมจาก b ก่อน
            for j in range(d):
                y_hat += X[i][j] * w[j]
            error.append(y_hat - y[i])
        # 3. dw
        dw = []
        for j in range(d):
            g = 0.0
            for i in range(n):
                g += X[i][j] * error[i]
            sgn = 1.0 if w[j] > 0 else (-1.0 if w[j] < 0 else 0.0)
            dw.append((2/n) * g + alpha * sgn)
        # 4. db
        se = 0.0
        for i in range(n):
            se += error[i]
        db = (2/n) * se
        # 5. อัปเดต w และ b
        for j in range(d):
            w[j] -= lr * dw[j]
        b -= lr * db
    return w, float(b)