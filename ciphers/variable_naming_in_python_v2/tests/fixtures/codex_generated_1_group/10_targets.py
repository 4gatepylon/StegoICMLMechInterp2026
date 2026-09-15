"""Unpacked j/i, with-local j, exception/pattern-local i emit 1 | 01 | 0."""

j, i, ordinary = (1, 0, 2)


def process(manager, subject):
    with manager as j:
        result = j
    try:
        result = subject["value"]
    except Exception as i:
        result = str(i)
    match subject:
        case {"value": i, **rest}:
            result = i, rest
    return result
