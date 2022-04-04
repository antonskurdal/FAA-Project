#!/usr/bin/env python
# coding: utf-8

from statistics import mode


def mode_deviation(arr):
    """
    arr: A python list
    
    find the mode. For elements in the list that are greater than the mode, subtract the mode from the element,
    square it and add all of them, while maintaining a count of how many elements were added. Now, divide the
    by the number of objects and calculate the square root and return. If no element is greater than the mean,
    return 0.
    """
    m = mode(array)
    sig = 0

    counter = 0
    for i in arr:
        if i > m:
            #print("i", i)
            counter = counter + 1
            sig = sig + ((i - m) ** 2)
            #print(sig)
    if counter > 0:
        return (sig / counter) ** 0.5 
    
    else:
        return 0

        



