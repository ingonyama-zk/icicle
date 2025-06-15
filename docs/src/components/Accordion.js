import React, { useState } from 'react';
import styles from './Accordion.module.css';

const AccordionItem = ({ title, children }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className={`${styles.item} ${isOpen ? styles.open : ''}`}>
      <button
        className={styles.header}
        onClick={() => setIsOpen(!isOpen)}
        aria-expanded={isOpen}
      >
        {title}
        <span className={styles.icon}>{isOpen ? '▴' : '▾'}</span>
      </button>
      <div className={styles.contentWrapper}>
        <div className={styles.content}>
          {children}
        </div>
      </div>
    </div>
  );
};

export default function Accordion() {
  return (
    <div className={styles.container}>
      <AccordionItem title="What is ICICLE?">
        ICICLE is a cryptography library designed to accelerate algorithms and protocols—starting with Zero-knowledge proofs—across diverse compute backends, including CPUs, GPUs, Apple Silicon, ZPU™ and more.
      </AccordionItem>

      <AccordionItem title="What problem does ICICLE solve?">
        It significantly accelerates core cryptographic operations like MSM, NTT, modular multiplication, and hashing on any backend.
      </AccordionItem>

        <AccordionItem title="Is ICICLE open-source?">
        <p>ICICLE is partially open-source:</p>
        <ul>
            <li>The core CUDA backend is source available under Ingonyama’s custom ICICLE-SA license.</li>
            <li>The frontend and language bindings (Rust & Go) are open-source under the MIT license.</li>
            <li>
            Full backend license details:{" "}
            <a href="https://dev.ingonyama.com/start/architecture/install_gpu_backend#licensing" target="_blank" rel="noopener noreferrer">
                ICICLE Licensing
            </a>.
            </li>
        </ul>
        </AccordionItem>

        <AccordionItem title="Which cryptographic primitives are supported?">
        <p>
            ICICLE supports a wide range of field, curve, vector, polynomial, MSM, NTT, FRI, and hashing operations across native, extension, and RNS fields. Learn more{" "}
            <a href="https://dev.ingonyama.com/start/architecture/libraries" target="_blank" rel="noopener noreferrer">
            here
            </a>.
        </p>
        </AccordionItem>

      <AccordionItem title="Which provers can integrate with ICICLE?">
        ICICLE is designed to integrate with virtually any prover. We've already integrated it into Gnark, Halo2 and also provide a native Groth16 prover in our icicle-snark repository.
      </AccordionItem>

        <AccordionItem title="What hardware is required to run ICICLE & which GPU models are recommended?">
        <p>
            For the CPU backend, any 64-bit CPU is fully supported. We've also successfully tested ICICLE on 32-bit CPUs, such as those found on Raspberry Pi, though with limited performance.
        </p>
        <p>
            For the CUDA backend, ICICLE runs on any NVIDIA GPU from the RTX 2060 and newer. We currently build for compute capabilities 7.5, 8.0, 8.6, and 8.9. For more details on GPU compatibility, see this{" "}
            <a href="https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/" target="_blank" rel="noopener noreferrer">
            reference
            </a>.
        </p>
        </AccordionItem>

      <AccordionItem title="Are non-Nvidia GPUs supported?">
        Metal (Apple GPUs) is partially supported but still experimental. Vulkan support is in progress and will expand compatibility in the future.
      </AccordionItem>

      <AccordionItem title="What programming languages does ICICLE support?">
        The core library is written in CUDA C++. Bindings exist for Rust and Go.
      </AccordionItem>

        <AccordionItem title="Does ICICLE support multi-GPU setups?">
        <p>
            Yes, ICICLE supports multi-GPU configurations. Learn more{" "}
            <a href="https://dev.ingonyama.com/api/rust-bindings/multi-gpu" target="_blank" rel="noopener noreferrer">
            here
            </a>.
        </p>
        </AccordionItem>

      <AccordionItem title="How much speedup can I expect using ICICLE?">
        Check out the Benchmark Dashboard to explore real-world results.
      </AccordionItem>

        <AccordionItem title="Who is using ICICLE today?">
        <p>
            ICICLE is used by leading projects and research teams across cryptography and infrastructure. See the full list{" "}
            <a href="https://dev.ingonyama.com/#ecosystem" target="_blank" rel="noopener noreferrer">
            here
            </a>.
        </p>
        </AccordionItem>

        <AccordionItem title="Where can I get help if I encounter issues?">
        <p>
            GitHub issues on the ICICLE repo are the primary support channel. Community discussions often happen in our Discord Server’s{" "}
            <a href="https://discord.gg/EVVXTdt6DF" target="_blank" rel="noopener noreferrer">
            ICICLE channel
            </a>.
        </p>
        </AccordionItem>

        <AccordionItem title="Can I contribute to ICICLE?">
        <p>
            Yes — contributions to ICICLE are welcome! You can submit pull requests or open issues on our{" "}
            <a href="https://github.com/ingonyama-zk/icicle" target="_blank" rel="noopener noreferrer">
            public GitHub repository
            </a>. If you're working on research or development aligned with ICICLE, consider applying to our{" "}
            <a href="https://www.ingonyama.com/post/ingonyama-research-grant-2025" target="_blank" rel="noopener noreferrer">
            Ingonyama Research Grant Program 2025
            </a>.
        </p>
        </AccordionItem>

        <AccordionItem title="What is the maximum MSM size tested?">
            <p>
                MSM sizes up to 2<sup>30</sup> have been tested successfully on the 3090Ti. Note that this applies only to certain backends. For example, Metal may not yet support streaming, which could limit maximum MSM size on that backend.
            </p>
        </AccordionItem>

        <AccordionItem title="What class of GPU is required to run ICICLE?">
        <p>
            Nvidia CUDA GPUs only. Some mobile/workstation GPUs (e.g. Quadro P520) may hit compilation issues that require patching CMake flags.
        </p>
        </AccordionItem>

        <AccordionItem title="Why does my benchmark report zero time per iteration?">
        <p>
            Likely due to LLVM optimization eliminating empty loops. Use <code>black_box</code> to prevent optimization during benchmarks.
        </p>
        </AccordionItem>

        <AccordionItem title="Is omega equal to root_of_unity?">
        <p>
            Yes, but ICICLE stores precomputed omegas for multiple powers. The specific omega used depends on the NTT size.
        </p>
        </AccordionItem>

        <AccordionItem title="Is there documentation for ntt_template_kernel_shared_rev?">
        <p>
            No formal docs. But the butterfly operation is here:{" "}
            <a
            href="https://github.com/ingonyama-zk/icicle/blob/main/icicle/appUtils/ntt/ntt.cuh#L153"
            target="_blank"
            rel="noopener noreferrer"
            >
            ntt.cuh L153
            </a>.
        </p>
        </AccordionItem>


    </div>
  );
}