import numpy as np
import torch


def tensor_to_ndarray(t: torch.Tensor) -> np.ndarray:
    """Convert a torch.Tensor to a NumPy array on CPU.

    Parameters
    ----------
    t:
        Input tensor.

    Returns
    -------
    np.ndarray
        Numpy array with the same data as ``t`` moved to CPU.
    """
    return np.array(t.cpu().tolist())
