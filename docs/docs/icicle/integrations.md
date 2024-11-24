# ICICLE integrated provers

ICICLE has been used by companies and projects such as [Celer Network](https://github.com/celer-network), [Consensys Gnark](https://github.com/Consensys/gnark), [EZKL](https://blog.ezkl.xyz/post/acceleration/), [ZKWASM](https://twitter.com/DelphinusLab/status/1762604988797513915) and others to accelerate their ZK proving pipeline.

Many of these integrations have been a collaboration between Ingonyama and the integrating company. We have learned a lot about designing GPU based ZK provers.

If you're interested in understanding these integrations better or learning how you can use ICICLE to accelerate your existing ZK proving pipeline this is the place for you.

## A primer to building your own integrations

Lets illustrate an ICICLE integration, so you can understand the core API and design overview of ICICLE.

![ICICLE architecture](./static/img/architecture-high-level.png)

Engineers usually use a cryptographic library to implement their ZK protocols. These libraries implement efficient primitives which are used as building blocks for the protocol; ICICLE is such a library. The difference is that ICICLE is designed from the start to run on GPUs; the Rust and Golang APIs abstract away all low level CUDA details. Our goal was to allow developers with no GPU experience to quickly get started with ICICLE.

A developer may use ICICLE with two main approaches in mind.

1. Drop-in replacement approach.
2. End-to-End GPU replacement approach.

The first approach for GPU-accelerating your Prover with ICICLE is quick to implement, but it has limitations, such as reduced memory optimization and limited protocol tuning for GPUs. It's a solid starting point, but those committed to fully leveraging GPU acceleration should consider a more comprehensive approach.

A End-to-End GPU replacement means performing the entire ZK proof on the GPU. This approach will reduce latency to a minimum and requires you to change the way you implement the protocol to be more GPU friendly. This approach will take full advantage of GPU acceleration. Redesigning your prover this way may take more engineering effort but we promise you that its worth it!

## Using ICICLE integrated provers

Here we cover how a developer can run existing circuits on ICICLE integrated provers.

### Gnark

[Gnark](https://github.com/Consensys/gnark) officially supports GPU proving with ICICLE. Currently only Groth16 on curve `BN254` is supported. This means that if you are currently using Gnark to write your circuits you can enjoy GPU acceleration without making many changes.

:::info

Currently ICICLE has been merged to Gnark [master branch](https://github.com/Consensys/gnark), however the [latest release](https://github.com/Consensys/gnark/releases/tag/v0.9.1) is from October 2023.

:::

Make sure your golang circuit project has `gnark` as a dependency and that you are using the master branch for now.

```
go get 	github.com/consensys/gnark@master
```

You should see two indirect dependencies added.

```
...
	github.com/ingonyama-zk/icicle v0.1.0 // indirect
	github.com/ingonyama-zk/iciclegnark v0.1.1 // indirect
...
```

:::info
As you may notice we are using ICICLE v0.1 here since golang bindings are only support in ICICLE v0.1 for the time being.
:::

To switch over to ICICLE proving, make sure to change the backend you are using, below is an example of how this should be done.

```
// toggle on
proofIci, err := groth16.Prove(ccs, pk, secretWitness, backend.WithIcicleAcceleration())

// toggle off
proof, err := groth16.Prove(ccs, pk, secretWitness)
```

Now that you have enabled `WithIcicleAcceleration` backend simple change the way your run your circuits to:

```
go run -tags=icicle main.go
```

Your logs should look something like this if everything went as expected.

```
13:12:05 INF compiling circuit
13:12:05 INF parsed circuit inputs nbPublic=1 nbSecret=1
13:12:05 INF building constraint builder nbConstraints=3
13:12:05 DBG precomputing proving key in GPU acceleration=icicle backend=groth16 curve=bn254 nbConstraints=3
13:12:05 DBG constraint system solver done nbConstraints=3 took=0.070259
13:12:05 DBG prover done acceleration=icicle backend=groth16 curve=bn254 nbConstraints=3 took=80.356684
13:12:05 DBG verifier done backend=groth16 curve=bn254 took=1.843888
```

`acceleration=icicle` indicates that the prover is running in acceleration mode with ICICLE.

You can reference the [Gnark docs](https://github.com/Consensys/gnark?tab=readme-ov-file#gpu-support) for further information.

### Halo2

[Halo2](https://github.com/zkonduit/halo2) fork integrated with ICICLE for GPU acceleration. This means that you can run your existing Halo2 circuits with GPU acceleration just by activating a feature flag.

To enable GPU acceleration just enable `icicle_gpu` [feature flag](https://github.com/zkonduit/halo2/blob/3d7b5e61b3052680ccb279e05bdcc21dd8a8fedf/halo2_proofs/Cargo.toml#L102).

This feature flag will seamlessly toggle on GPU acceleration for you.
