"""Closure reads reuse outer j; shadow j is distinct: 1 | 1 | 0."""


def outer():
    j = 1

    def inner():
        return j

    def shadow():
        j = 2
        return j

    return inner(), shadow()


def payload():
    i = 0
    return i
