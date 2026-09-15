"""Two disjoint groups interleave in source order; see expected_decodes.json."""

import math as j
import os.path  # noqa: F401 -- An ordinary import binding must remain visible.


class Box:
    a = 0
    b = 2

    def method(self):
        # Bare j resolves globally in a method; self.j is an attribute.
        return j, self.j, "i j"


def i():
    return 0
