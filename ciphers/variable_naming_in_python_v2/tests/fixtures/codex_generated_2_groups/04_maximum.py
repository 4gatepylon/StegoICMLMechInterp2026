"""Two disjoint groups interleave in source order; see expected_decodes.json."""


def first():
    j = 1
    return j


def second():
    b = 2
    return b


def third():
    j = 3
    return j


def fourth():
    a = 4
    return a


def fifth():
    i = 5
    return i


def sixth():
    b = 6
    return b


def seventh():
    j = 7
    return j
