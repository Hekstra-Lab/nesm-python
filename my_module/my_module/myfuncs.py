import numpy as np

def normalize(v):
    """
    Divide a 1D vector by its length.
    """
    mag = np.sqrt(np.sum(v**2))
    return v/mag
