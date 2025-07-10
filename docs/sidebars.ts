export default {
  GettingStartedSidebar: [
    {
      type: "category",
      label: "Introduction",
      collapsed: false,
      items: [
        {
          type: "doc",
          id: "start/intro/start",
        },
        {
          type: "doc",
          id: "start/intro/getting_started",
        },
      ],
    },
    {
      type: "category",
      label: "Architecture",
      collapsed: false,
      items: [
        {
          type: "doc",
          id: "start/architecture/arch_overview",
        },
        {
          type: "doc",
          id: "start/architecture/build_your_own_backend",
        },
        {
          type: "doc",
          id: "start/architecture/install_gpu_backend",
        },
        {
          type: "doc",
          id: "start/architecture/libraries",
        },
        {
          type: "doc",
          id: "start/architecture/multi-device",
        },
      ],
    },
    {
      type: "category",
      label: "Programmer's Guide",
      collapsed: false,
      items: [
        {
          type: "doc",
          id: "start/programmers_guide/general",
        },
        {
          type: "doc",
          id: "start/programmers_guide/build_from_source",
        },
        {
          type: "doc",
          id: "start/programmers_guide/cpp",
        },
        {
          type: "doc",
          id: "start/programmers_guide/go",
        },
        {
          type: "doc",
          id: "start/programmers_guide/rust",
        },
      ],
    },
    {
      type: "category",
      label: "Integration & Support",
      collapsed: false,
      items: [
        {
          type: "doc",
          id: "start/integration-&-support/contributor-guide"
        },
        {
          type: "doc",
          id: "start/integration-&-support/faq_and_troubleshooting"
        },
        {
          type: "doc",
          id: "start/integration-&-support/grants",
        },  
                {
          type: "doc",
          id: "start/integration-&-support/integrations",
        }, 
        {
          type: "doc",
          id: "start/integration-&-support/colab-instructions",
        },  
        {
          type: "doc",
          id: "start/integration-&-support/migrate_from_v2",
        },
        {
          type: "doc",
          id: "start/integration-&-support/migrate_from_v3",
        },
        {
          type: "doc",
          id: "start/integration-&-support/benchmarks",
        },
      ],
    },
  ],

  apisidebar: [
    {
      type: "doc",
      label: "API Overview",
      id: "api/overview",
    },
    {
      type: "category",
      label: "C++",
      collapsed: false,
      items: [
        {
          type: "doc",
          label: "C++ Overview",
          id: "api/cpp/cppstart",
        },
        {
          type: "doc",
          label: "MSM",
          id: "api/cpp/msm",
        },
        {
          type: "doc",
          label: "NTT / ECNTT",
          id: "api/cpp/ntt",
        },
        {
          type: "doc",
          label: "Vector operations",
          id: "api/cpp/vec_ops",
        },
        {
          type: "doc",
          label: "Program",
          id: "api/cpp/program",
        },
        {
          type: "doc",
          label: "Polynomials",
          id: "api/cpp/polynomials/overview",
        },
        {
          type: "doc",
          label: "Hash",
          id: "api/cpp/hash",
        },
        {
          type: "doc",
          label: "Merkle-Tree",
          id: "api/cpp/merkle",
        },
        {
          type: "doc",
          label: "Sumcheck",
          id: "api/cpp/sumcheck",
        },
        {
          type: "doc",
          label: "FRI",
          id: "api/cpp/fri",
        },
        {
          type: "doc",
          label: "Pairings",
          id: "api/cpp/pairings",
        },
        {
          type: "doc",
          label: "PQC ML-KEM",
          id: "api/cpp/lattice/pqc_ml_kem",
        },
        {
          type: "doc",
          label: "Serialization",
          id: "api/cpp/serialization",
        },
        {
          type: "doc",
          label: "Matrix Operations",
          id: "api/cpp/matrix_ops",
        },
      ],
    },
    {
      type: "category",
      label: "Golang Bindings",
      collapsed: false,
      items: [
        {
          type: "doc",
          label: "Golang Overview",
          id: "api/golang-bindings/golang-bindings",
        },
        {
          type: "doc",
          label: "MSM",
          id: "api/golang-bindings/msm",
        },
        {
          type: "doc",
          label: "MSM pre-computation",
          id: "api/golang-bindings/msm-pre-computation",
        },
        {
          type: "doc",
          label: "NTT",
          id: "api/golang-bindings/ntt",
        },
        {
          type: "doc",
          label: "EC-NTT",
          id: "api/golang-bindings/ecntt",
        },
        {
          type: "doc",
          label: "Vector operations",
          id: "api/golang-bindings/vec-ops",
        },
        {
          type: "doc",
          label: "Multi GPU Support",
          id: "api/golang-bindings/multi-gpu",
        },
        {
          type: "doc",
          label: "Hash",
          id: "api/golang-bindings/hash",
        },
        {
          type: "doc",
          label: "Merkle-Tree",
          id: "api/golang-bindings/merkle",
        },
        {
          type: "doc",
          label: "PQC ML-KEM",
          id: "api/golang-bindings/lattice/pqc-ml-kem",
        },
      ],
    },
    {
      type: "category",
      label: "Rust Bindings",
      collapsed: false,
      items: [
        {
          type: "doc",
          label: "Rust Overview",
          id: "api/rust-bindings/rust-bindings",
        },
        {
          type: "doc",
          label: "MSM",
          id: "api/rust-bindings/msm",
        },
        {
          type: "doc",
          label: "NTT",
          id: "api/rust-bindings/ntt",
        },
        {
          type: "doc",
          label: "ECNTT",
          id: "api/rust-bindings/ecntt",
        },
        {
          type: "doc",
          label: "Vector operations",
          id: "api/rust-bindings/vec-ops",
        },
        {
          type: "doc",
          label: "Program",
          id: "api/rust-bindings/program",
        },
        {
          type: "doc",
          label: "Polynomials",
          id: "api/rust-bindings/polynomials",
        },
        {
          type: "doc",
          label: "Hash",
          id: "api/rust-bindings/hash",
        },
        {
          type: "doc",
          label: "Merkle-Tree",
          id: "api/rust-bindings/merkle",
        },
        {
          type: "doc",
          label: "Sumcheck",
          id: "api/rust-bindings/sumcheck",
        },
        {
          type: "doc",
          label: "FRI",
          id: "api/rust-bindings/fri",
        },
        {
          type: "doc",
          label: "PQC ML-KEM",
          id: "api/rust-bindings/lattice/pqc-ml-kem",
        },
        {
          type: "doc",
          label: "Serialization",
          id: "api/rust-bindings/serialization",
        },
        {
          type: "doc",
          label: "Pairings",
          id: "api/rust-bindings/pairing",
        },  
        {
          type: "doc",
          label: "Matrix Operations",
          id: "api/rust-bindings/matrix_ops",
        },
      ],
    },
  ],
};
