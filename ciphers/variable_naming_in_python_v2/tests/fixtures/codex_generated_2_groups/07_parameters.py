"""Two disjoint groups interleave in source order; see expected_decodes.json."""

j = 1


def transform(a, b=j):
    return a, (lambda i: i + b)(0)
