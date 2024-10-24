"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Union

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    """Negates each element of the input tensor and propagates the gradient."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Returns the negation of the input tensor."""
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Returns the negation of the gradient."""
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    """Computes the element-wise inverse of the input tensor and its gradient."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Returns the inverse of the input tensor."""
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Returns the gradient for the inverse operation."""
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    """Computes element-wise addition of two tensors and propagates the gradient."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Returns the element-wise sum of two tensors."""
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns the gradient for both inputs of the addition."""
        return grad_output, grad_output


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Union[Tensor, int, None]) -> Tensor:
        """Return 1 if all are true"""
        if dim is not None:
            # Fix: Check if dim is Tensor before calling .item()
            dim_val = int(dim.item()) if isinstance(dim, Tensor) else int(dim)
            return a.f.mul_reduce(a, dim_val)
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


# TODO: Implement for Task 2.3.


class Mul(Function):
    """Multiplies two tensors element-wise and propagates the gradient."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Returns the element-wise product of two tensors."""
        ctx.save_for_backward(a, b)
        return a.f.mul_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns the gradient for multiplication."""
        a, b = ctx.saved_values
        return grad_output * b, grad_output * a


class Sigmoid(Function):
    """Applies the sigmoid function to the input tensor and propagates the gradient."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Returns the sigmoid of the input tensor."""
        sigmoid_t1 = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(sigmoid_t1)
        return sigmoid_t1

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Returns the gradient for the sigmoid function."""
        sigmoid_x = ctx.saved_values[0]
        ones = sigmoid_x.zeros(sigmoid_x.shape)
        ones._tensor._storage[:] = [1.0] * len(ones._tensor._storage)
        return grad_output * sigmoid_x * (ones - sigmoid_x)


class ReLU(Function):
    """Applies the ReLU function to the input tensor and propagates the gradient."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Returns the ReLU of the input tensor."""
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Returns the gradient for the ReLU function."""
        input_tensor = ctx.saved_values[0]
        out = input_tensor.zeros(input_tensor.shape)
        for i, val in enumerate(input_tensor._tensor._storage):
            out._tensor._storage[i] = 1.0 if val > 0.0 else 0.0
        return grad_output * out


class Log(Function):
    """Computes the logarithm of the input tensor and propagates the gradient."""

    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Returns the logarithm of the input tensor."""
        ctx.save_for_backward(a)
        return a.f.log_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Returns the gradient for the logarithm function."""
        input_tensor = ctx.saved_values[0]
        return grad_output / input_tensor


class Exp(Function):
    """Computes the exponential of the input tensor and propagates the gradient."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Returns the exponential of the input tensor."""
        exp_t1 = t1.f.exp_map(t1)
        ctx.save_for_backward(exp_t1)
        return exp_t1

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Returns the gradient for the exponential function."""
        exp_x = ctx.saved_values[0]
        return grad_output * exp_x


class Sum(Function):
    """Computes the sum of a tensor along a specific dimension and propagates the gradient."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Union[Tensor, int]) -> Tensor:
        """Returns the sum of the input tensor along the specified dimension."""
        if isinstance(dim, int):
            dim_val = dim
        else:
            dim_val = int(dim.item())
        ctx.save_for_backward(a.shape, dim_val)
        return a.f.add_reduce(a, dim_val)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns the gradient for the sum operation."""
        input_shape, dim = ctx.saved_values
        grad_input = grad_output
        while grad_input.dims < len(input_shape):
            grad_input = grad_input.view(*([1] + list(grad_input.shape)))

        new_shape = list(input_shape)
        new_shape[dim] = 1
        grad_input = grad_input.view(*new_shape)
        grad_input = grad_input + zeros(input_shape)

        # Return Tensor instead of float for the second element
        return grad_input, zeros(())  # Return a scalar tensor instead of 0.0


class LT(Function):
    """Compares two tensors element-wise and propagates the gradient."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Returns the element-wise comparison of two tensors."""
        return a.f.lt_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns zero gradients for the less-than operation."""
        return 0.0 * grad_output, 0.0 * grad_output


class EQ(Function):
    """Compares two tensors for equality element-wise and propagates the gradient."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Returns the element-wise equality of two tensors."""
        return a.f.eq_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns zero gradients for the equality operation."""
        return 0.0 * grad_output, 0.0 * grad_output


class IsClose(Function):
    """Checks if two tensors are element-wise close and propagates the gradient."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Returns the element-wise closeness of two tensors."""
        return a.f.is_close_zip(a, b)


class Permute(Function):
    """Permutes the dimensions of a tensor according to the given order and propagates the gradient."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, *order: int) -> Tensor:
        """Returns the tensor with permuted dimensions."""
        order_ints = tuple(int(o.item()) if hasattr(o, "_tensor") else o for o in order)
        ctx.save_for_backward(order_ints)
        return a._new(a._tensor.permute(*order_ints))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, ...]:
        """Returns the gradient for the permuted tensor."""
        order = ctx.saved_values[0]
        inv_order = [0] * len(order)
        for i, p in enumerate(order):
            inv_order[p] = i
        grad_input = grad_output.permute(*inv_order)
        # Fix: Convert the zeros to Tensors
        return (grad_input,) + tuple(zeros(()) for _ in order)


class View(Function):
    """Reshapes a tensor without changing its data and propagates the gradient."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Returns the tensor with a new shape."""
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Returns the gradient for the reshaped tensor."""
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Computes the central difference approximation of the gradient for a given function."""

    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
