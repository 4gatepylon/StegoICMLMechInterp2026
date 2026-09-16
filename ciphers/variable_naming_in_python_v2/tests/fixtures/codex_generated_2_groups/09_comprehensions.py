"""Two disjoint groups interleave in source order; see expected_decodes.json."""

j = 0


def size(a):
    return a


def outer():
    j = 1

    def update():
        nonlocal j
        j += 1

    def touch():
        global j
        j += 1

    values = [(i := b) for b in range(2)]
    return j, i, values, update, touch
