
# Merkle Tree API Documentation

## What is a Merkle Tree?

A **Merkle tree** is a cryptographic data structure that allows for **efficient verification of data integrity**. It consists of:
- **Leaf nodes**, each containing a piece of data.
- **Internal nodes**, which store the **hashes of their child nodes**, leading up to the **root node** (the cryptographic commitment).

With ICICLE, you have the **flexibility** to build various tree topologies based on your needs. The user must define:

1. **Hasher per layer** ([Link to Hasher API](./hash.md)) with a **default input size**.
2. **Size of a leaf element** (in bytes): This defines the **granularity** of the data used for opening proofs.

### Tree Structure

It is assumed that the **root node** is a single node. The **height of the tree** is determined by the **number of layers** specified by the user.  
Each layer has its own **arity**, calculated as:

$$
{arity}_i = \frac{layers[i].inputSize}{layer[i-1].outputSize}
$$

and for layer 0:

$$
{arity}_0 = \frac{layers[0].inputSize}{leafSize}
$$



**TODO:**  
- Add an **image** describing the tree structure.  
- Show **code snippets** for defining the tree.

---

## Padding

:::note  
This feature is not yet supported in **v3.1** and will be available in **v3.2**.  
:::

When the input for **layer 0** is smaller than expected, ICICLE can apply **padding** to align the data.

**Padding Schemes:**
1. **Zero padding:** Adds zeroes to the remaining space.
2. **Repeat last leaf:** The final leaf element is repeated to fill the remaining space.

**TODO:**  
- Add **code snippet** for configuring padding.

---

## Root as Commitment

The **root of the Merkle tree** acts as a **cryptographic commitment** to the entire dataset.  
With the **root hash**, a verifier can confirm the presence of specific data using a **Merkle proof**, without requiring access to the entire dataset.

**TODO:**  
- Add **image** of the root and its role in commitments.  
- Add **code snippet** to demonstrate how to retrieve the root.

---

## Proof of Inclusion via Merkle Paths

A **Merkle path** is a collection of **sibling hashes** that allows the verifier to **reconstruct the root hash** from a specific leaf.  
This enables anyone with the **path and root** to verify that the **leaf** belongs to the committed dataset.

**TODO:**  
- Add an **updated image** showing a Merkle path.  
- Add **code snippet** for generating a proof.

---

## Pruned vs. Full Paths

1. **Full Path:**  
   - Contains all **sibling nodes** and **intermediate hashes** from the leaf to the root.

2. **Pruned Path:**  
   - Includes only the **necessary sibling hashes**, excluding intermediate nodes that can be recomputed during verification.

**TODO:**  
- Add **images** to show the difference between full and pruned paths.  
- Include a **code snippet** demonstrating each path type.

---

## Verifying Merkle Proofs

To **verify a Merkle proof**, the verifier:

1. Uses the **leaf** and the provided **path** to **recompute the hashes** along the path.
2. If the **computed root** matches the provided commitment, the **proof is valid**.

:::note  
The verification **reuses the Merkle tree** since it needs to know the hash functions and configurations, which are defined by the tree itself.  
:::

**TODO:**  
- Add **code snippet** for verifying a proof.

---

## Handling Partial Tree Storage

In cases where the **Merkle tree is large**, only the **top layers** may be stored to conserve memory.  
When verifying, the **first layers** (closest to the leaves) are **recomputed dynamically**.

### Recomputing Layers During Verification

- The verifier recomputes **missing layers** as needed.
- Verification continues from the **first stored layer** to the **root**.

**TODO:**  
- Add **code snippet** for initializing the tree when using **partial storage**.

---

## Conclusion

This documentation provides a comprehensive overview of the **Merkle Tree API** and how to build, configure, and verify trees using ICICLE. With **customizable hash functions**, **layer-by-layer control**, and support for **pruned and full paths**, ICICLE enables efficient cryptographic verification tailored to your needs.

**Next Steps:**  
Once familiar with the concepts, explore:
- **Advanced hashing techniques** with ICICLE.
- **Optimizing memory usage** for large trees.
- **Using ICICLE with different backends**, such as **CPU** and **CUDA**.
