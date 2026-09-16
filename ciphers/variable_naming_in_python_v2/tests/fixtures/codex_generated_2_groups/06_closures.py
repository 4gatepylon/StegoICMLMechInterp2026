"""Two disjoint groups interleave in source order; see expected_decodes.json."""


def outer():
    j = 1

    def size(a):
        return a

    def inner():
        return j

    def shadow():
        b = 2
        return b

    return size, inner(), shadow()


def payload():
    i = 0
    return i
