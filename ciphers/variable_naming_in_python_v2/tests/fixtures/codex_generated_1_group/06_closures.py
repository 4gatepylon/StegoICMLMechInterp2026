"""Closure reads reuse outer j; shadow j is distinct: 1 | 01 | 0."""


def outer():
    j = 1

    def size(i):
        return i

    def inner():
        return j

    def shadow():
        j = 2
        return j

    return size, inner(), shadow()


def payload():
    i = 0
    return i
