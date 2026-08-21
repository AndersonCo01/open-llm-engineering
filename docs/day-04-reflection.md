# Day 4 Reflection

## 1. Token IDs versus embeddings

Why does an LLM use embedding vectors instead of token IDs directly?

My answer:

Token IDs are only arbitrary integer labels; their numerical values do not represent meaning. Embeddings convert each token into a learned vector that can capture semantic and grammatical relationships. These continuous vectors can be processed effectively by a neural network.

## 2. Embedding shapes

What shapes enter and leave `nn.Embedding` in this lesson?

My answer:

In this lesson, `nn.Embedding` receives token IDs with shape:

```text
[batch_size, context_length]
```

It returns embedding vectors with shape:

```text
[batch_size, context_length, embedding_dimension]
```

Each token ID is replaced by one embedding vector.

## 3. Overlapping context

Why do context windows overlap?

My answer:

Context windows overlap so the model can learn from every possible adjacent input-target relationship and preserve information across window boundaries. Without overlap, many relationships between nearby tokens would be excluded from the training data.

## 4. Position information

Why are position embeddings necessary?

My answer:

Token embeddings represent what the tokens are, but not where they appear. Position embeddings provide information about token order, allowing the model to distinguish sequences such as "dog bites man" and "man bites dog."

## 5. Broadcasting

How does broadcasting help when adding position embeddings?

My answer:

Position embeddings usually have shape:

```text
[context_length, embedding_dimension]
```

Token embeddings have shape:

```text
[batch_size, context_length, embedding_dimension]
```

Broadcasting automatically applies the same position embeddings to every sequence in the batch, so we do not need to copy them manually for each example.

## 6. Pooling information loss

What information is lost by mean pooling the sequence axis?

My answer:

Mean pooling averages all token vectors into one vector. This removes token order, individual token positions, and many token-specific details. It preserves only a general summary of the sequence, so different sequences can potentially produce similar pooled representations.
