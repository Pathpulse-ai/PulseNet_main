def autopad(k, p=None, d=1):
    """
    k: kernel size (int)
    p: padding size (int) or None
    d: dilation (int)
    """
    # If p is not provided, calculate 'same' padding for given kernel size and dilation
    if p is None:
        p = (k - 1) * d // 2
    return p