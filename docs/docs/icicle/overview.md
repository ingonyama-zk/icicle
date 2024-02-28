# What is ICICLE?

[![Static Badge](https://img.shields.io/badge/Latest-v1.4.0-8a2be2)](https://github.com/ingonyama-zk/icicle/releases)

![Static Badge](https://img.shields.io/badge/Machines%20running%20ICICLE-544-lightblue)



[ICICLE](https://github.com/ingonyama-zk/icicle) is a cryptography library for ZK using GPUs. ICICLE implements blazing fast cryptographic primitives such as EC operations, MSM, NTT, Poseidon hash and more on GPU.

ICICLE allows developers with minimal GPU experience to effortlessly accelerate their ZK application; from our experiments, even the most naive implementation may yield 10X improvement in proving times.

ICICLE has been used by many leading ZK companies such as [Celer Network](https://github.com/celer-network), [Gnark](https://github.com/Consensys/gnark) and others to accelerate their ZK proving pipeline.

## Dont have access to a GPU?

We understand that not all developers have access to a GPU and we don't want this to limit anyone from developing with ICICLE.
Here are some ways we can help you gain access to GPUs:

### Grants

At Ingonyama we are interested in accelerating the progress of ZK and cryptography. If you are an engineer, developer or an academic researcher we invite you to checkout [our grant program](https://www.ingonyama.com/blog/icicle-for-researchers-grants-challenges). We will give you access to GPUs and even pay you to do your dream research!

### Google Colab

This is a great way to get started with ICICLE instantly. Google Colab offers free GPU access to a NVIDIA T4 instance, it's acquired with 16 GB of memory which should be enough for experimenting and even prototyping with ICICLE.

For an extensive guide on how to setup Google Colab with ICICLE refer to [this article](./colab-instructions.md).

If none of these options are appropriate for you reach out to us on [telegram](https://t.me/RealElan) we will do our best to help you.

### Vast.ai

[Vast.ai](https://vast.ai/) is a global GPU marketplace where you can rent many different types of GPUs by the hour for [competitive pricing](https://vast.ai/pricing). They provide on-demand and interruptible rentals depending on your need or use case; you can learn more about their rental types [here](https://vast.ai/faq#rental-types).

:::note

If none of these options suit your needs, contact us on [telegram](https://t.me/RealElan) for assistance. We're committed to ensuring that a lack of a GPU doesn't become a bottleneck for you. If you need help with setup or any other issues, we're here to do our best to help you.

:::

## What can you do with ICICLE?

[ICICLE](https://github.com/ingonyama-zk/icicle) can be used in the same way you would use any other cryptography library. While developing and integrating ICICLE into many proof systems, we found some use case categories:

### Circuit developers

If you are a circuit developer and are experiencing bottlenecks while running your circuits, an ICICLE integrated prover may be the solution.

ICICLE has been integrated into a number of popular ZK provers including [Gnark prover](https://github.com/Consensys/gnark) and [Halo2](https://github.com/zkonduit/halo2). This means that you can enjoy GPU acceleration for your existing circuits immediately without writing a single line of code by simply switching on the GPU prover flag!

### Integrating into existing ZK provers

From our collaborations we have learned that its possible to accelerate a specific part of your prover to solve for a specific bottleneck.

ICICLE can be used to accelerate specific parts of your prover without completely rewriting your ZK prover.

### Developing your own ZK provers

If your goal is to build a ZK prover from the ground up, ICICLE is an ideal tool for creating a highly optimized and scalable ZK prover. A key benefit of using GPUs with ICICLE is the ability to scale your ZK prover efficiently across multiple machines within a data center.

### Developing proof of concepts

ICICLE is also ideal for developing small prototypes. ICICLE has Golang and Rust bindings so you can easily develop a library implementing a specific primitive using ICICLE. An example would be develop a KZG commitment library using ICICLE.
