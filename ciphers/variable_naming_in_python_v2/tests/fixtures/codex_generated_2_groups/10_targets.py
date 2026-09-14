"""Two disjoint groups interleave in source order; see expected_decodes.json."""

j, i, ordinary = (1, 0, 2)


def process(manager, subject):
    with manager as b:
        result = b
    try:
        result = subject["value"]
    except Exception as a:
        result = str(a)
    match subject:
        case {"value": a, **rest}:
            result = a, rest
    return result
