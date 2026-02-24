---
title: "Language Models with Belief State Generalizes Much Better during Pre-training"
date: 2025-11-26
author: "Sunny Sanyal"
editor: "Dr. Mike Mozer"
#summary: "A small proof-of-concept study on integrating differentiable belief states into GPT-style decoders."
tags:
  - language models
  - belief state
  - pre-training
  - Transformer
draft: true
---

> *We introduce “belief state” in transformer blocks which serves as a persistent memory that carries information across time and sequences. GPT-style decoder language models augmented with belief state generalizes much better during pre-training.*

## Abstract

Large language models typically operate as stateless next-token predictors: each forward pass depends only on the input sequence and the model weights. In this post, I explore a simple proof-of-concept (PoC) that augments a decoder-only transformer with a **belief state**—a compact persistent vector that evolves during inference and training.

The guiding idea is that a model should be able to *maintain* and *update* internal hypotheses about the sequence it is processing, similar to how humans keep track of intermediate thoughts. I outline how the belief state is initialized, updated, and integrated into the residual stream, and summarize preliminary behavior observed during validation and comprehensive perplexity evaluation.

## Introduction

Modern decoder-only transformers excel at next-token prediction, yet they perform all reasoning within a single forward pass. Despite massive capacity, they remain *memory-less* across forward passes: they cannot maintain an evolving internal representation unless the token context explicitly reencodes it.

Several recent research threads—pondering, test-time compute, iterative refinement, latent recurrence—hint at the value of giving models a *persistent* internal state. This post builds on those themes and sketches a minimal, practical approach that fits neatly inside a GPT-2–style codebase.

Here, a **belief state** is a learned vector or tuple of vectors that:

- represents the model’s current internal hypothesis,
- gets updated after every forward pass,
- optionally warms up over a few passes before evaluation,
- affects the token-level predictions for the remainder of the sequence.

This post documents the design choices, implementation shape, and initial observations from integrating such a state into the training loop.

## Method

### 1. Belief State Definition

We attach to the model a compact state:

```
state ∈ ℝ^{S × d}
```

where `S` is the number of slots (often 1–4) and `d` matches the model embedding dimension.

### 2. Injecting the State

At each layer ℓ, we inject the belief state into the residual stream:

```
h_ℓ ← h_ℓ + W_ℓ · state
```

### 3. Updating the State

After each forward pass, the model updates the state:

```
state ← f_θ(state, h_last)
```

EMA-style updates typically work well:

```
state_new = (1 − α) · state + α · g(h_last)
```

### 4. Warmup Passes

Before computing val loss or comprehensive perplexity, we allow warmup passes:

```
for t in 1..T:
    forward(x, state)
    update(state)
```

### 5. Evaluation Protocol

- **Val loss**: deterministic, non-overlapping chunks; fresh state; optional warmup  
- **Comprehensive eval**: sequential, stride=0; fresh state; optional warmup

## Closing Thoughts

This is a small step toward train-time and test-time recurrence inside transformers. Even modest belief states can subtly shift model behavior and improve consistency.

