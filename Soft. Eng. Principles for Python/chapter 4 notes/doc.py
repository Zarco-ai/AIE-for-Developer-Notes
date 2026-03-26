# Docstrings are what python outputs whenever a user calls help on your docstring!

#Example
def function(x):
    """High level description of function,
    Additional details on function.
    
    :param x: description of parameter x
    :return: description of return value
    
    >>> # Example function usage, 
    Expected output of example function usage
    """
    # Function Code
    
help(function) #This will output the docstring of my function

import this #This returns a copy of "The Zen of Python" - all about creating readable code

"""Testing Code"""
#Use 'doctest' to test docstring in your module