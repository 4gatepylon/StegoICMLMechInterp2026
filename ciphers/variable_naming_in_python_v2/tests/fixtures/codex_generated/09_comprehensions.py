"""Module/outer j and outer walrus i encode 110; comprehension j trails."""

j = 0


def outer():
    j = 1

    def update():
        nonlocal j
        j += 1

    def touch():
        global j
        j += 1

    values = [(i := j) for j in range(2)]
    return j, i, values, update, touch
