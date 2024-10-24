"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(a: float, b: float) -> float:
    """Multiply two numbers.

    Args:
    ----
        a: First number.
        b: Second number.

    Returns:
    -------
        The product of a and b.

    """
    return a * b


def id(x: float) -> float:
    """Identity function.

    Args:
    ----
        x: Input value.

    Returns:
    -------
        The input value x.

    """
    return x


def add(a: float, b: float) -> float:
    """Add two numbers.

    Args:
    ----
        a: First number.
        b: Second number.

    Returns:
    -------
        The sum of a and b.

    """
    return a + b


def neg(x: float) -> float:
    """Negate a number.

    Args:
    ----
        x: Input value.

    Returns:
    -------
        The negation of x.

    """
    return -x


def lt(a: float, b: float) -> float:
    """Less than comparison.

    Args:
    ----
        a: First number.
        b: Second number.

    Returns:
    -------
        True if a is less than b, False otherwise.

    """
    return 1.0 if a < b else 0.0


def eq(a: float, b: float) -> float:
    """Equality comparison.

    Args:
    ----
        a: First number.
        b: Second number.

    Returns:
    -------
        True if a is equal to b, False otherwise.

    """
    return 1.0 if a == b else 0.0


def max(a: float, b: float) -> float:
    """Maximum of two numbers.

    Args:
    ----
        a: First number.
        b: Second number.

    Returns:
    -------
        The larger of a and b.

    """
    return a if a > b else b


def is_close(a: float, b: float) -> float:
    """Check if two numbers are close to each other.

    Args:
    ----
        a: First number.
        b: Second number.

    Returns:
    -------
        True if the absolute difference between a and b is less than 1e-2, False otherwise.

    """
    return (a - b < 1e-2) and (b - a < 1e-2)


def sigmoid(x: float) -> float:
    """Sigmoid activation function.

    Args:
    ----
        x: Input value.

    Returns:
    -------
        The sigmoid of x.

    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """ReLU activation function.

    Args:
    ----
        x: Input value.

    Returns:
    -------
        The ReLU of x.

    """
    return x if x > 0.0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Natural logarithm.

    Args:
    ----
        x: Input value.

    Returns:
    -------
        The natural logarithm of x.

    """
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Exponential function.

    Args:
    ----
        x: Input value.

    Returns:
    -------
        e raised to the power of x.

    """
    return math.exp(x)


def inv(x: float) -> float:
    """Inverse function.

    Args:
    ----
        x: Input value.

    Returns:
    -------
        The reciprocal of x.

    """
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    """Backward pass for logarithm.

    Args:
    ----
        x: Input value.
        d: Gradient from the next layer.

    Returns:
    -------
        The gradient with respect to x.

    """
    return d / (x + EPS)


def inv_back(x: float, d: float) -> float:
    """Backward pass for inverse function.

    Args:
    ----
        x: Input value.
        d: Gradient from the next layer.

    Returns:
    -------
        The gradient with respect to x.

    """
    return (-1.0 / x**2) * d


def relu_back(x: float, d: float) -> float:
    """Backward pass for ReLU activation.

    Args:
    ----
        x: Input value.
        d: Gradient from the next layer.

    Returns:
    -------
        The gradient with respect to x.

    """
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Apply a function to each element in a list.

    Args:
    ----
        fn: Function to apply to each element.
        lst: Input list.

    Returns:
    -------
        A new list with the function applied to each element.

    """

    def _map(lst: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in lst:
            ret.append(fn(x))
        return ret

    return _map


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher order function that takes two lists and combines them element-wise using a given function.

    Args:
    ----
        fn: Function to combine elements.

    Returns:
    -------
        Function that takes two lists and combines them element-wise using the given function.

    """

    def _zipWith(lst1: Iterable[float], lst2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(lst1, lst2):
            ret.append(fn(x, y))
        return ret

    return _zipWith


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Higher order function that reduces a list to a single value using a given function.

    Args:
    ----
        fn: Function to combine elements.
        start: starting value

    Returns:
    -------
       Function that takes a list and reduces it to a single value using the given function.

    """

    def _reduce(lst: Iterable[float]) -> float:
        result = start
        for x in lst:
            result = fn(result, x)
        return result

    return _reduce


def negList(lst: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list.

    Args:
    ----
        lst: Input list.

    Returns:
    -------
        A new list with all elements negated.

    """
    return map(neg)(lst)


def addLists(lst1: Iterable[float], lst2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists.

    Args:
    ----
        lst1: First input list.
        lst2: Second input list.

    Returns:
    -------
        A new list with elements added element-wise.

    """
    return zipWith(add)(lst1, lst2)


def sum(lst: Iterable[float]) -> float:
    """Sum all elements in a list.

    Args:
    ----
        lst: Input list.

    Returns:
    -------
        The sum of all elements in the list.

    """
    return reduce(add, 0.0)(lst)


def prod(lst: Iterable[float]) -> float:
    """Calculate the product of all elements in a list.

    Args:
    ----
        lst: Input list.

    Returns:
    -------
        The product of all elements in the list.

    """
    return reduce(mul, 1.0)(lst)
