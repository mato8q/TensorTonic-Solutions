def ridge_regression(X, y, lr, epochs, alpha):
    """
    Perform ridge regression using gradient descent.
    Returns: tuple of (weights_list, bias)
    """
    X =np.array(X, dtype=np.float64)
    y =np.array(y, dtype=np.float64)
    n,d= X.shape
    n = float(n)
    w= np.zeros(d,dtype=np.float64)
    b = 0.0
    lr = float(lr)
    alpha =float(alpha)

    for _ in range(epochs):
        y_hat = np.dot(X,w)+b
        error = y_hat - y

        dw = (2.0/n)*(np.dot(X.T,error)) + 2.0*alpha*w
        db = float((2.0/n)*np.sum(error))

        w = w- lr*dw
        b = float(b- lr*db)


    weights_list = np.round(w,4).tolist() 
    bias = round(float(b),4)

    return ([round(float(x), 4) for x in w], round(b, 4))
    