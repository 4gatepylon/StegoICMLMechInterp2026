"""Module j, parameters i/j, lambda i emit 1 | 01 | 0; default j is global."""

j = 1


def transform(i, j=j):
    return i, (lambda i: i + j)(0)
