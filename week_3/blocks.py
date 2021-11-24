import numpy as np

#######################################################
# put `sigmoid_forward` and `sigmoid_grad_input` here #
#######################################################

def sigmoid_forward(x_input):
    """sigmoid nonlinearity
    # Arguments
        x_input: np.array of size `(n_objects, n_in)`
    # Output
        the output of relu layer
        np.array of size `(n_objects, n_in)`
    """
    # h = linearforward(x_input,W,b)
    # assume x_input is the value from h(x)
    output = 1/(1+np.exp(-x_input))
    return output

def sigmoid_grad_input(x_input, grad_output):
    """sigmoid nonlinearity gradient. 
        Calculate the partial derivative of the loss 
        with respect to the input of the layer
    # Arguments
        x_input: np.array of size `(n_objects, n_in)` # x_input is the linear mapping
        grad_output: np.array of size `(n_objects, n_in)` 
            dL / df
    # Output
        the partial derivative of the loss 
        with respect to the input of the function
        np.array of size `(n_objects, n_in)` 
        dL / dh
    """
    # asuume that x_input is the f(h) sigmoid linear function
    
    df_dh = sigmoid_forward(x_input) * (1-sigmoid_forward(x_input))
    grad_input = grad_output * df_dh
    
    return grad_input

#######################################################
#      put `nll_forward` and `nll_grad_input` here    #
#######################################################

def nll_forward(target_pred, target_true):
    """Compute the value of NLL
        for a given prediction and the ground truth
    # Arguments
        target_pred: predictions - np.array of size `(n_objects, 1)`
        target_true: ground truth - np.array of size `(n_objects, 1)`
    # Output
        the value of NLL for a given prediction and the ground truth
        scalar
    """

    output = - np.mean(target_true*np.log(target_pred) + (1-target_true)*np.log(1-target_pred))
    return output

def nll_grad_input(target_pred, target_true):
    """Compute the partial derivative of NLL
        with respect to its input
    # Arguments
        target_pred: predictions - np.array of size `(n_objects, 1)`
        target_true: ground truth - np.array of size `(n_objects, 1)`
    # Output
        the partial derivative 
        of NLL with respect to its input
        np.array of size `(n_objects, 1)`
    """
    grad_input = np.mean((target_pred - target_true)/ (target_pred*(1-target_pred)))
    return grad_input

