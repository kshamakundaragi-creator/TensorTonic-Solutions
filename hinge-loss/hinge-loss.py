import numpy as np

def hinge_loss(y_true, y_score, margin=1, reduction="mean"):
    
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    
    if y_true.shape != y_score.shape:
        raise ValueError("Shapes must match")
    
    if not np.all(np.isin(y_true, [-1, 1])):
        raise ValueError("Labels must be -1 or +1")
    
    loss = np.maximum(0, margin - y_true * y_score)
    
    if reduction == "mean":
        return float(np.mean(loss))
    elif reduction == "sum":
        return float(np.sum(loss))
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")