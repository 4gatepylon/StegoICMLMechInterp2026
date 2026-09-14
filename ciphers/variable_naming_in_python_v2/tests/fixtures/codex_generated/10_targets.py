"""Unpacked module j, with-local j, and exception/pattern-local i emit 110."""

j, ordinary = (1, 2)


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
