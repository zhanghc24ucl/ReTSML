
def ewm(x, alpha=None, halflife=None, span=None):
    from numpy import exp, log, empty_like
    if alpha is not None:
        pass
    elif halflife is not None:
        alpha = 1 - exp(-log(2) / halflife)
    elif span is not None:
        alpha = 2 / (span + 1)
    else:
        raise ValueError("Must provide one of alpha, halflife, or span")

    rv = empty_like(x)
    for j in range(rv.shape[1]):
        rv[0, j] = x[0, j]
        for i in range(1, rv.shape[0]):
            rv[i, j] = alpha * x[i, j] + (1 - alpha) * rv[i - 1, j]
    return rv
