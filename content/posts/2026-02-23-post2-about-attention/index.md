---
title: "Attention Is All You Need — But Do We Really Know What It's Doing?"
date: 2026-02-23
author: "Sunny Sanyal"
editor: "Dr. Mike Mozer"
tags:
  - attention
  - Transformer
  - self-attention
  - deep learning
  - language models
draft: false
---

> *Attention is the most celebrated mechanism in modern deep learning, yet most practitioners treat it as a black box. This post strips it down to first principles, builds intuition from scratch, and honestly confronts some of its messier realities.*

## Abstract

The attention mechanism sits at the heart of virtually every state-of-the-art model in natural language processing, vision, and multimodal learning. In this post, I try to do two things at once: give a ground-up derivation that someone with a linear-algebra background could follow without prior transformer experience, and then complicate the rosy picture with some genuinely puzzling empirical observations. The goal is not just to understand *how* attention works, but to start asking sharper questions about *why* it works so well—and where it might be silently failing us.

## Introduction

I remember the first time I read the original "Attention is All You Need" paper. My reaction was something like: *is that it?* The core computation fits on half a page, the math is mostly matrix multiplications and a softmax, and the architecture diagram looks almost too clean. It felt like something important was being hidden from me.

Several years and a lot of training runs later, I still think something is being hidden—but now I suspect it is hidden from the *researchers* too, not just from beginners. We have an increasingly sophisticated vocabulary for describing what attention *does*, but a surprisingly thin account of *why* it generalizes the way it does, *what* the heads actually learn to compute, and *when* the whole thing is likely to break.

This post is my attempt to build up a careful first-principles account, flag the places where the standard story is incomplete, and point toward some research directions that I find genuinely exciting.

## Background

### Sequence modeling before attention

Before transformers took over, recurrent architectures dominated sequence modeling. LSTMs and GRUs process tokens one at a time, updating a hidden state vector at each step. The hidden state has to compress everything the model knows about the past into a fixed-size vector—a bottleneck that becomes especially painful for long sequences.

Attention was actually introduced *inside* recurrent models before transformers existed. The key insight from Bahdanau et al. (2015) was simple: instead of forcing the decoder to read off a single context vector, let it *look back* at all the encoder hidden states and weight them dynamically. The weights—the "attention"—are learned, input-dependent, and can focus on whatever part of the source sequence matters most for the current prediction step.

The transformer paper generalized this to the point where the recurrence itself becomes unnecessary. If you can attend over the full sequence in every layer, you do not need to thread information through time step by step. Every position can, in principle, talk directly to every other position.

### What problem is attention solving?

It is worth being precise here. Attention solves a *routing* problem. The representation of token $i$ at layer $\ell$ needs to incorporate information from some subset of other tokens. The question is: which ones, and how much of each?

A convolutional layer answers this question with a fixed local receptive field—useful but inflexible. A dense linear layer answers it with fixed global weights—flexible but incapable of adapting to the specific content of the input. Attention answers it with *content-dependent, dynamically computed* weights. That is the core value proposition.

## Method

### Scaled dot-product attention: a careful derivation

Let me work through the computation step by step, because the notation in most papers glosses over details that matter for intuition.

We start with a sequence of $n$ token representations, each a vector in $\mathbb{R}^d$. Stack them into a matrix $X \in \mathbb{R}^{n \times d}$.

We project $X$ through three separate learned linear maps:

```
Q = X W_Q        # queries:  n × d_k
K = X W_K        # keys:     n × d_k
V = X W_V        # values:   n × d_v
```

where $W_Q, W_K \in \mathbb{R}^{d \times d_k}$ and $W_V \in \mathbb{R}^{d \times d_v}$.

The attention weights are then computed as:

```
A = softmax( Q K^T / sqrt(d_k) )    # n × n
```

And the output is:

```
Out = A V                             # n × d_v
```

The figure below shows the full flow—tokens go in, get split into Q, K, V, the dot products are scored and softmaxed, and the values are aggregated.

![Scaled dot-product attention: queries, keys, and values flow through the computation to produce attended outputs.](images/scaled-dot-product-attention.png)

Let's unpack what each piece is doing.

**Queries and Keys.** The query $q_i = x_i W_Q$ represents what token $i$ is *looking for*. The key $k_j = x_j W_K$ represents what token $j$ *has to offer*. Their dot product $q_i \cdot k_j$ measures how relevant token $j$'s content is to token $i$'s current representational need. Softmax turns the raw scores into a probability distribution over the sequence.

**The $\sqrt{d_k}$ scaling.** This is often mentioned but rarely explained well. The issue is that dot products of random unit vectors in high dimensions concentrate around zero, but the *variance* of the dot product scales with $d_k$. Without scaling, the softmax input can have very large magnitude, pushing the distribution toward near-one-hot, which kills gradient flow (the softmax saturates and gradients vanish). Dividing by $\sqrt{d_k}$ keeps variance roughly constant regardless of dimension.

**Values.** The value $v_j = x_j W_V$ is the actual content that gets mixed in if token $j$ receives high attention weight. Crucially, the keys/queries and values are *separate projections*. The decision of *who to attend to* is decoupled from *what information to extract*. This is a subtle but important design choice.

### Multi-head attention

A single attention head produces a single weighted mix of value projections. Multi-head attention runs $H$ attention heads in parallel, each with its own $W_Q^h, W_K^h, W_V^h$, then concatenates and projects:

```
MultiHead(X) = Concat(head_1, ..., head_H) W_O
```

where each $\text{head}_h = \text{Attention}(X W_Q^h, X W_K^h, X W_V^h)$.

The motivating idea is that different heads can specialize. One head might track syntactic dependencies, another coreference, another positional proximity. Whether this specialization actually happens in practice is a more complicated story—I will come back to it.

![Multi-head attention runs several attention heads in parallel and concatenates their outputs before a final linear projection.](images/multi-head-attention.png)

A practical note: in standard implementations, we split the model dimension $d$ evenly across heads, so $d_k = d / H$. Increasing the number of heads does not increase the total parameter count of the Q, K, V projections; it just changes how the capacity is partitioned.

### Positional encodings and their discontents

Bare attention is *permutation equivariant*: if you shuffle the input tokens, the output shuffles the same way. This is sometimes presented as a feature—the model is "flexible about order"—but for most language tasks, order matters enormously. We need to inject positional information somehow.

The original transformer used fixed sinusoidal encodings added to the input embeddings. Later work moved to learned absolute positions. More recently, relative position encodings (RPE) and rotary position embeddings (RoPE) have become dominant because they generalize better to sequence lengths longer than those seen during training.

RoPE is particularly elegant. Rather than adding a position-dependent bias to the input, it rotates the query and key vectors in pairs of dimensions. The dot product of a rotated query and key then depends only on their *relative* position, not their absolute positions. This means the model can extrapolate to longer sequences—at least in principle.

![RoPE rotates query and key vectors by position-dependent angles so that their dot product encodes relative distance between tokens.](images/rope-positional-encoding.png)

### Causal masking

For decoder-only language models, we need *causal* (autoregressive) attention: token $i$ can only attend to tokens at positions $\leq i$. This is implemented by adding a large negative constant to the attention logits at illegal positions before the softmax:

```
A = softmax( Q K^T / sqrt(d_k) + M )
```

where $M_{ij} = 0$ if $j \leq i$ and $M_{ij} = -\infty$ otherwise. The $-\infty$ entries become zero after softmax, effectively masking out future tokens.

![The causal attention mask is a lower-triangular matrix: each token can attend to itself and all earlier tokens, but not to future ones.](images/causal-attention-mask.png)

This is simple in principle but has practical implications. During training on long sequences, the early tokens in a context window are attended to by *every subsequent token*, while the last token attends to the whole window. This creates very unequal gradient flow, which partly explains why models sometimes show degraded performance toward the end of very long contexts.

## Empirical Observations Worth Thinking About

This is where I want to push beyond the standard exposition.

### Attention heads are mostly redundant

Multiple studies have found that a large fraction of attention heads in trained transformers can be pruned (set to uniform or zero attention) with minimal effect on performance. Voita et al. (2019) found that in BERT-style models, only a handful of heads are truly "important" for most tasks, with the rest being essentially inert.

This raises a question: if multi-head attention is supposed to encourage specialization and diverse routing, why does so much of the capacity sit unused? One hypothesis is that the model uses redundancy as a form of robustness—the same information is computed multiple ways so that if the training signal is noisy, at least one head gets it right. Another is that the heads are not actually as specialized as we tend to imagine, and the benefit comes from an ensemble-like averaging effect.

![Visualizing which attention heads survive structured pruning: most are dropped with near-zero effect on held-out perplexity.](images/attention-head-pruning.png)

### The attention pattern is not the computation

A persistent misconception is that visualizing the attention weights—the matrix $A$—tells you what the model is "paying attention to" in a meaningful sense. This is not quite right.

The output of an attention head is $AV$. The matrix $A$ tells you *how much* each position contributes to each output, but it does not tell you *what* that contribution is. Two heads with identical attention patterns but different value matrices compute entirely different things. Conversely, a head that attends uniformly might still compute something meaningful if the values are well-structured.

Jain & Wallace (2019) showed that attention weights are not reliable explanations for model predictions in the sense of faithfully reflecting what information was used. This created a cottage industry of follow-up work, with the field landing roughly here: attention is *one* piece of evidence about information routing, but not a complete picture.

![Raw attention weight heatmaps for several heads in a 12-layer transformer. The patterns look interpretable—until you check whether perturbing them actually changes the output.](images/attention-weight-heatmap.png)

### Induction heads and in-context learning

One of the cleanest mechanistic results in the interpretability literature is the discovery of *induction heads* by Olsson et al. (2022). These are pairs of attention heads that implement a specific algorithm: if the sequence contains `[A][B]...[A]`, the second head copies `B` to the current position. Together, they implement a kind of fuzzy pattern matching over the context.

The remarkable finding is that the emergence of induction heads coincides with a sharp drop in training loss—a "phase transition"—and is strongly correlated with in-context learning ability. Models without functioning induction heads show nearly no in-context learning.

![Animation of the induction head circuit: a prefix-matching head identifies the previous occurrence of the current token; a copying head then moves the token that followed it.](images/induction-head-circuit.gif)

This is one of the few cases where we have a fairly complete story: a specific circuit, with a specific function, whose presence explains a specific capability. It suggests that the full transformer computation might eventually be decomposable into a relatively small number of such circuits—but we are nowhere close to that for larger models.

### Length generalization: the open wound

Training transformers on short sequences and deploying on longer ones is notoriously unreliable. This is true even with RoPE, despite its theoretical advantages. Models often show surprisingly sharp degradation at lengths just beyond the training maximum.

The underlying issue seems to be that the attention distribution changes character at long range. At sequence length $n$, the softmax output is a distribution over $n$ elements. As $n$ grows, maintaining a sharp, peaked distribution requires logits that scale with $\log n$. Standard dot-product attention does not have any mechanism to ensure this scaling, so the attention entropy can drift in unpredictable ways for long contexts.

Several fixes have been proposed—ALiBi, YaRN, length-scaled attention—but none of them feels fully principled. This is an area where I think the community is still missing a clean theoretical understanding.

![Perplexity as a function of sequence length for models trained at various context windows. The cliff at the training cutoff is real and still not fully explained.](images/length-generalization.png)

## Closing Thoughts

There is something humbling about spending years working with a mechanism and still finding it genuinely surprising. Attention is simple enough to write in a few lines of code, but rich enough to underlie models that do things we do not fully understand.

What I have come to believe is that the value of attention is not so much in any single clever trick—not the dot product, not the scaling, not the multi-head structure—but in the way all of these pieces combine to create a *differentiable routing mechanism* that can be trained end-to-end at scale. The specific form matters less than the fact that it is flexible, expressive, and stable to train.

The open problems that excite me most are on the mechanistic side: can we build interpretable accounts of what attention is doing in large models, the way Olsson et al. did for induction heads? And on the architectural side: are there routing mechanisms that generalize better to long sequences, or that waste less capacity on redundant heads?

I do not have tidy answers to either question. But I think asking them precisely is already most of the work.

---

*Corrections and pushback are welcome. Some of this reflects my own read of a fast-moving literature, and I'm sure I've gotten things wrong in places.*
