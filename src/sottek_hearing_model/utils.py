# -*- coding: utf-8 -*-
"""
utils.py
------------

Utility functions for use in the RefMap project.

Requirements
------------


Ownership and Quality Assurance
-------------------------------
Author: Mike JB Lotinga (m.j.lotinga@edu.salford.ac.uk)
Institution: University of Salford
 
Date created: 03/09/2025
Date last modified: 03/09/2025
Python version: 3.11

Copyright statement: This file and code is subject to licence as detailed
in the code repository
(https://github.com/acoustics-code-salford/refmap-psychoacoustics)

"""

from math import copysign

def roundTrad(val, digits=3):
    """
    Return a rounded value to a specified number of decimal places using the traditional approach (.5 rounds up).

    Parameters
    ----------
    
    val : float
          The value to be rounded.

    digits : int
             The number of decimal places to round to (default=3).
    
    Returns
    -------

     : float
          The rounded value.

    """
    return round(val+10**(-len(str(val))-1), digits)


def displayRndVal(val, digits=3, floor=True):
    """
    Return a string representation of a value rounded to a specified number of decimal places.

    Parameters
    ----------

    val : float
          The value to be rounded.
    
    digits : int
             The number of decimal places to round to (default=3).

    floor : bool
            If True, the string will represent the value as "<" or ">" signed 1/10**digits

    Returns
    -------

    valString : str
                The rounded value as a string.
        
    """
    
    crit = 1/10**digits

    if abs(val) < crit and floor:
        if val > 0:
            valString = "<" + str(copysign(crit, val))
        else:
            valString = ">" + str(copysign(crit, val))

    else:
        valRnd = roundTrad(val, digits)
        valString = f'{valRnd:f}'.rstrip("0")
        dec = int(valString.split('.')[-1])
        zs = len(valString.split('.')[-1]) - len(valString.split('.')[-1].lstrip("0"))
        nzeros = digits - len(str(dec)) - zs
        if nzeros > 0:
            valString = valString + nzeros*"0"

    return(valString)