"""Separate bindings emit 1 | 01 | 0; ordinary bindings remain visible."""


def control(j):
    return j


def size(i, j):
    return i + j


def solve(value):
    i = value + 1
    return i
