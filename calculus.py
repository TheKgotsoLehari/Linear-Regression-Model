import numpy as np 

def model(x, m):
    """
    Compute the predicted values f(x) = m * x.
    
    Parameters:
    x (numpy array): The input data.
    m (float): The model parameter (slope).
    
    Returns:
    numpy array: The predicted values.
    """
    return m * x

def error_function(x, y, m):
    """
    Compute the mean squared error (J(m)).
    
    Parameters:
    x (numpy array): The input data.
    y (numpy array): The actual values.
    m (float): The model parameter (slope).
    
    Returns:
    float: The mean squared error.
    """
    predictions = model(x, m)
    errors = predictions - y
    mse = np.mean(errors**2) / 2
    return mse

def derivative(x, y, m):
    """
    Compute the derivative of the error function with respect to m (dJ/dm).
    
    Parameters:
    x (numpy array): The input data.
    y (numpy array): The actual values.
    m (float): The model parameter (slope).
    
    Returns:
    float: The derivative of the error function.
    """
    predictions = model(x, m)
    errors = predictions - y
    grad = np.mean(errors * x)
    return grad

def update_step(m, learning_rate, gradient):
    """
    Update the model parameter using gradient descent.
    
    Parameters:
    m (float): The current model parameter (slope).
    learning_rate (float): The learning rate.
    gradient (float): The gradient of the error function with respect to m.
    
    Returns:
    float: The updated model parameter.
    """
    return m - learning_rate * gradient

if __name__ == "__main__":
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])

    m = 0
    learning_rate = 0.01
    epochs = 1000

    for epoch in range(epochs):
        grad = derivative(x, y, m)
        m = update_step(m, learning_rate, grad)
        if epoch % 100 == 0:
            cost = error_function(x, y, m)
            print(f'Epoch {epoch}: Cost = {cost:.4f}, m = {m:.4f}')
