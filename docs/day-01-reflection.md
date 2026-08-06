# Day 1 Reflection

## 1. Shape versus rank

A tensor's rank is the total number of dimensions, or axes, it has. Its shape describes the number of elements along each dimension.

Rank tells us how many axes the data has, while shape tells us the size of each axis.

For example, a tensor with shape `[2, 3, 4]` has rank `3`.

## 2. Matrix compatibility

A `[2, 3]` matrix can multiply a `[3, 4]` matrix because their inner dimensions match: `3 == 3`.

A `[2, 3]` matrix cannot multiply a `[2, 4]` matrix because their inner dimensions do not match: `3 != 2`.

## 3. Resulting shape

Multiplying a `[2, 3]` matrix by a `[3, 4]` matrix using Python's `@` operator produces a tensor with shape `[2, 4]`.

The matching inner dimensions are used during the calculation. The outer dimensions determine the result: the number of rows from the first matrix and the number of columns from the second matrix.

The general rule is:

`[a, b] @ [b, c] → [a, c]`

## 4. Zero vectors

A zero vector is invalid for cosine similarity because its magnitude is zero. The cosine-similarity formula divides by the product of the two vector magnitudes, which would cause division by zero.

A zero vector also has no defined direction, so it cannot be meaningfully compared by direction.

## 5. Sequence averaging

In a tensor with shape `[batch, sequence, features]`, averaging dimension `1` summarizes the sequence positions, such as tokens or time steps.

This operation changes the shape from `[batch, sequence, features]` to `[batch, features]`.