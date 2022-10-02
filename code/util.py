import math

# This files contains some common utilities function

# Given two numbers x, y, find the integers between them inclusive
def int_between(x, y):
    if x < y:
        return range(math.ceil(x), math.floor(y) + 1)
    else:
        return range(math.ceil(y), math.floor(x) + 1)