"""Import j, class-body i/j, and definition i emit 1 | 01 | 0."""

import math as j
import os.path  # noqa: F401 -- An ordinary import binding must remain visible.


class Box:
    i = 0
    j = 2

    def method(self):
        # Bare j resolves globally in a method; self.j is an attribute.
        return j, self.j, "i j"


def i():
    return 0
