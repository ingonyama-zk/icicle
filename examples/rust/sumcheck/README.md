# Sumcheck example

This example, instantiates a R1CS sumcheck instance with the parameters 
* `Sample size = 1<<22` field: BN254 scalar.
* Sumcheck IOP is part of a more general protocol, we use merlin to "simulate" a previous state of the transcript, and generate a seed randomness based on the previous state.
* The verifier also simulates the previous state of the transcript using Merlin to generate an identical seed randomness.
* In this example, we configured the Fiat Shamir protocol to run with the blake3 hash. Experiment with different hashes such as Keccak, blake2s , Poseidon etc. 
* Experiment with different sizes, fields, etc.
* The code runs in both CPU/GPU in a device agnostic manner. It assumes that the cuda backend is installed in the `icicle/backend/cuda/` folder, defaults to CPU in the absence of a device backend.

Running the example with script
```sh
# for CPU
./run.sh -d CPU
# for CUDA
./run.sh -d CUDA -b /path/to/cuda/backend/install/dir
# for METAL (not supported yet)
./run.sh -d METAL -b /path/to/cuda/backend/install/dir
```

Sample outputs: M1 mac
```rust
[2025-02-17T20:57:25Z INFO  sumcheck] Generate e,A,B,C of log size 22, time 954.447375ms
[2025-02-17T20:57:26Z INFO  sumcheck] Compute claimed sum time 439.200917ms
[2025-02-17T20:57:28Z INFO  sumcheck] Prover time 2.383280625s
Valid proof!
[2025-02-17T20:57:28Z INFO  sumcheck] verify time 201.666µs
[2025-02-17T20:57:28Z INFO  sumcheck] total time 3.777288875s
```

Sample outputs: Ubuntu Desktop 22.04 128GB RAM	GPU: RTX 4080
```rust
[WARNING] Defaulting to Ingonyama icicle-cuda-license-server at `5053@license.icicle.ingonyama.com`. For more information about icicle-cuda-license, please contact support@ingonyama.com.
[2025-02-17T21:10:33Z INFO  sumcheck] Generate e,A,B,C of log size 22, time 890.615763ms
[2025-02-17T21:10:33Z INFO  sumcheck] Compute claimed sum time 374.166708ms
[2025-02-17T21:10:34Z INFO  sumcheck] Prover time 224.319171ms
Valid proof!
[2025-02-17T21:10:34Z INFO  sumcheck] verify time 183.244µs
[2025-02-17T21:10:34Z INFO  sumcheck] total time 1.489436224s
```

