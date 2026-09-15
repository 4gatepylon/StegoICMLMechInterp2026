"""Repeated writes and loop iterations emit each binding once: 1 | 01 | 0."""


def control():
    j = 0
    j += 1
    return j


def size(i):
    j = 1
    for j in range(2):
        ordinary = j
    return i, j, ordinary


def payload():
    i = 0
    i += 1
    return i
