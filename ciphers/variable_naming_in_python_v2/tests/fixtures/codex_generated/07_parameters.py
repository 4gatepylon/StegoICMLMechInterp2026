"""Module j, parameter j, lambda i emit 1 | 1 | 0; default j is global."""

j = 1


def transform(j=j):
    return (lambda i: i + j)(0)
