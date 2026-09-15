"""Module j, size i, outer j/walrus i emit 1 | 01 | 0; comprehension j trails."""

j = 0


def size(i):
    return i


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
