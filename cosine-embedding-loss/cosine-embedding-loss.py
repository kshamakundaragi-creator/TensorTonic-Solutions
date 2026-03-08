import numpy as np
def cosine_embedding_loss(x1, x2, label, margin):
    x1 = np.array(x1)
    x2 = np.array(x2)
    dot = np.dot(x1,x2)
    norm1 = np.linalg.norm(x1)
    norm2 = np.linalg.norm(x2)
    cosine_sim = dot/(norm1*norm2)
    if label == 1:
      loss = 1-cosine_sim
    else:
        loss = max(0, cosine_sim - margin)
    return (loss)