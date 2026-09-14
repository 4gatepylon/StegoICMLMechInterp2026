"""Repeated writes and loop iterations still emit only j, j, i: 1 | 1 | 0."""


def control():
    j = 0
    j += 1
    return j


def size():
    j = 1
    for j in range(2):
        ordinary = j
    return j, ordinary


def payload():
    i = 0
    i += 1
    return i
