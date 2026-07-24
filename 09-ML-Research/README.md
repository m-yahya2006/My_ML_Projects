# Attention Mechanism — Paper Reproduction

## Paper

**Attention Is All You Need**
*Vaswani et al., 2017*

## What I Implemented

I implemented the **Scaled Dot-Product Attention** from the Transformer paper using PyTorch.

The implementation:

* calculates the attention scores,
* converts the scores into attention weights using Softmax,
* and produces the final output using the Value matrix.

## The Math

The attention formula is:

Attention(Q, K, V) = Softmax((Q × Kᵀ) / √dk) × V

Where:

* **Q (Query):** What is this token looking for?
* **K (Key):** What information does this token have?
* **V (Value):** What information should this token share?

## Why do we divide by (√dk)?

The value √dk is called the scaling factor. It prevents the attention scores from becoming too large. This helps the Softmax function stay stable during training. Without √dk, the scores become very large, causing Softmax to give almost all the attention to a few values. As a result, the gradients become very small, making learning slower or even stopping it.

Dividing by (√dk):

* keeps the scores at a reasonable size,
* makes Softmax more stable,
* and helps the model learn better during training.

## Results

The implementation successfully:

* calculated the attention scores,
* applied the scaling factor,
* converted the scores into attention weights,
* generated the final output,
* and produced the expected tensor shapes.

## What I Learned

From this implementation, I learned:

* what Query, Key, and Value represent,
* why the Key matrix must be transposed before matrix multiplication,
* why we divide by (\sqrt{d_k}),
* why Softmax is applied on the last dimension (`dim=-1`),
* how attention weights show the importance of each token,
* and how attention helps each token collect useful information from the other tokens in the sequence.
