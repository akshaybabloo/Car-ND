import numpy as np


def softmax(x):
    r"""
    The equation is given by

    .. math::

        S(y_{i})=\frac{e^{y_{i}}}{\sum_{j}^{n}e^{y_{j}}}

    Parameters
    ----------
    x

    Returns
    -------

    """
    return np.divide(np.exp(x), np.sum(np.exp(x), axis=0))

logits = [3.0, 1.0, 0.2]

print(softmax(logits))
