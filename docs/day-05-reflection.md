# Day 5 Reflection

## 1. Query, key, and value roles

What different roles do queries, keys, and values play?

My answer:

Queries represent what the current token is looking for. Keys represent what
each token offers or how it can be matched. Values contain the information that
will be retrieved. The query is compared with all keys, and the resulting
scores determine how much of each value contributes to the output.

## 2. Score scaling

Why are query-key scores divided by `sqrt(head_dimension)`?

My answer:

Query-key dot products tend to grow as the head dimension increases. Dividing
by `sqrt(head_dimension)` keeps the scores at a manageable scale. Without this
scaling, softmax may become extremely peaked, producing very small gradients
and making training unstable.

## 3. Mask before softmax

Why must the causal mask be applied before softmax?

My answer:

The mask must be applied before softmax so future-token scores can be changed
to negative infinity. Softmax then converts those scores into exactly zero
attention weight. If masking happened after softmax, future tokens would
already have affected the normalization and taken probability away from valid
tokens.

## 4. Weight rows

What does one row of the attention-weight matrix represent?

My answer:

One row represents how a particular query token distributes its attention
across all available key tokens. For example, row `i` shows how much token `i`
attends to each token in the sequence. In causal attention, it can attend only
to itself and earlier tokens.

## 5. Normalized weights

Why must every row of attention weights sum to one?

My answer:

Softmax converts every row of attention scores into a probability distribution.
Therefore, the weights are nonnegative and sum to one. This allows the
attention output to be a weighted combination of the value vectors and makes
the weights easier to interpret as relative importance.

## 6. Attention versus bigrams

How is self-attention more powerful than the Day 3 bigram model?

My answer:

A bigram model predicts the next token using only the immediately preceding
token. Self-attention can use information from every permitted token in the
context window. It can therefore:

- capture long-range relationships;
- focus on different tokens depending on context;
- produce context-dependent representations; and
- model relationships between words that are far apart.

For example, when predicting a verb, self-attention can connect it to a subject
several tokens earlier, while a bigram model cannot directly use that distant
information.
