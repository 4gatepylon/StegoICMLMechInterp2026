"""Two disjoint groups interleave in source order; see expected_decodes.json."""


def control():
    j = 0
    j += 1
    return j


def size(a):
    b = 1
    for b in range(2):
        ordinary = b
    return a, b, ordinary


def payload():
    i = 0
    i += 1
    return i
