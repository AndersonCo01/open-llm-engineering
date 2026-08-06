"""Day 1: tensor fundamentals.

A tensor is a rectangular collection of numbers. Its shape describes the size
of every axis. LLMs represent tokens, model parameters, and intermediate
calculations as tensors.
"""

import torch
from torch import Tensor


def guided_examples() -> None:
    """Create several tensors and print the information needed to inspect them."""
    scalar = torch.tensor(7.0)
    vector = torch.tensor([1.0, 2.0, 3.0])
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    print(f"scalar: {scalar}; shape={scalar.shape}; dimensions={scalar.ndim}")
    print(f"vector: {vector}; shape={vector.shape}; dimensions={vector.ndim}")
    print(f"matrix:\n{matrix}\nshape={matrix.shape}; dimensions={matrix.ndim}")

    weights = torch.tensor([[0.2, 0.8], [0.5, 0.5]])
    values = torch.tensor([[10.0, 0.0], [0.0, 20.0]])
    mixed_values = weights @ values
    print(f"matrix multiplication result:\n{mixed_values}")

def matrix_multiply(left: Tensor, right: Tensor) -> Tensor:
    """Return the matrix product of two rank-2 tensors.

    Raise ValueError with a helpful message for invalid ranks or shapes.
    """
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("Both tensors must have two dimensions")

    if left.shape[1] != right.shape[0]:
        raise ValueError("Tensor shapes are incompatible")

    return left @ right


def cosine_similarity(first: Tensor, second: Tensor) -> Tensor:
    """Calculate cosine similarity between two one-dimensional vectors."""
    if first.ndim != 1 or second.ndim != 1:
        raise ValueError("Both tensors must be one-dimensional vectors")

    if first.shape != second.shape:
        raise ValueError("Vectors must have the same shape")

    dot_product = torch.dot(first, second)

    first_norm = torch.linalg.vector_norm(first)
    second_norm = torch.linalg.vector_norm(second)

    if first_norm == 0 or second_norm == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector")

    return dot_product / (first_norm * second_norm)




def batch_mean(batch: Tensor) -> Tensor:
    """Average a [batch, sequence, features] tensor over the sequence axis."""
    if batch.ndim != 3: 
        raise ValueError("Batch tensor must have three dimensions")
    
    return batch.mean(dim=1)

if __name__ == "__main__":
    guided_examples()

