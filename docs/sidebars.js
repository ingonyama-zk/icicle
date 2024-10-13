module.exports = {
  GettingStartedSidebar: [
    {
      type: "doc",
      label: "Introduction",
      id: "introduction",
    },
    {
      type: "category",
      label: "ICICLE",
      link: {
        type: `doc`,
        id: 'icicle/overview',
      },
      collapsed: false,
      items: [
        {
          type: "category",
          label: "Getting started",
          link: {
            type: `doc`,
            id: "icicle/getting_started",
          },
          collapsed: false,
          items: [
            {
              type: "doc",
              label: "Build ICICLE from source",
              id: "icicle/build_from_source",
            },
            {
              type: "category",
              label: "Programmers guide",
              link: {
                type: `doc`,
                id: "icicle/programmers_guide/general",
              },
              collapsed: false,
              items: [
                {
                  type: "doc",
                  label: "C++",
                  id: "icicle/programmers_guide/cpp",
                },
                {
                  type: "doc",
                  label: "Rust",
                  id: "icicle/programmers_guide/rust",
                },
                {
                  type: "doc",
                  label: "Go",
                  id: "icicle/programmers_guide/go",
                }
              ],
            },
          ],
        },
        {
          type: "category",
          label: "Architecture overview",
          link: {
            type: `doc`,
            id: "icicle/arch_overview"
          },
          collapsed: false,
          items: [
            {
              type: "doc",
              label: "CUDA Backend",
              id: "icicle/install_cuda_backend"
            },
            {
              type: "doc",
              label: "Multi-Device Support",
              id: "icicle/multi-device",
            },
            {
              type: "doc",
              label: "Build Your Own Backend",
              id: "icicle/build_your_own_backend"
            },
          ]
        },
        {
          type: "doc",
          label: "ICICLE libraries",
          id: "icicle/libraries",
        },
        {
          type: "category",
          label: "Compute API",
          link: {
            type: `doc`,
            id: 'icicle/primitives/overview',
          },
          collapsed: true,
          items: [
            {
              type: "doc",
              label: "MSM",
              id: "icicle/primitives/msm",
            },
            {
              type: "doc",
              label: "NTT / ECNTT",
              id: "icicle/primitives/ntt",
            },
            {
              type: "doc",
              label: "Vector operations",
              id: "icicle/primitives/vec_ops",
            },
            {
              type: "doc",
              label: "Polynomials",
              id: "icicle/polynomials/overview",
            },
            {
              type: "doc",
              label: "Hash",
              id: "icicle/primitives/hash",
            },
            {
              type: "category",
              label: "Golang bindings",
              link: {
                type: `doc`,
                id: "icicle/golang-bindings",
              },
              collapsed: true,
              items: [
                {
                  type: "category",
                  label: "MSM",
                  link: {
                    type: `doc`,
                    id: "icicle/golang-bindings/msm",
                  },
                  collapsed: true,
                  items: [
                    {
                      type: "doc",
                      label: "MSM pre computation",
                      id: "icicle/golang-bindings/msm-pre-computation",
                    }
                  ]
                },
                {
                  type: "doc",
                  label: "NTT",
                  id: "icicle/golang-bindings/ntt",
                },
                {
                  type: "doc",
                  label: "EC-NTT",
                  id: "icicle/golang-bindings/ecntt",
                },
                {
                  type: "doc",
                  label: "Vector operations",
                  id: "icicle/golang-bindings/vec-ops",
                },
                {
                  type: "doc",
                  label: "Multi GPU Support",
                  id: "icicle/golang-bindings/multi-gpu",
                },
              ]
            },
            {
              type: "category",
              label: "Rust bindings",
              link: {
                type: `doc`,
                id: "icicle/rust-bindings",
              },
              collapsed: true,
              items: [
                {
                  type: "doc",
                  label: "MSM",
                  id: "icicle/rust-bindings/msm",
                },
                {
                  type: "doc",
                  label: "NTT",
                  id: "icicle/rust-bindings/ntt",
                },
                {
                  type: "doc",
                  label: "ECNTT",
                  id: "icicle/rust-bindings/ecntt",
                },
                {
                  type: "doc",
                  label: "Vector operations",
                  id: "icicle/rust-bindings/vec-ops",
                },
                {
                  type: "doc",
                  label: "Polynomials",
                  id: "icicle/rust-bindings/polynomials",
                },
                {
                  type: "doc",
                  label: "Hash",
                  id: "icicle/rust-bindings/hash",
                },
                // {
                //   type: "doc",
                //   label: "Multi GPU Support (TODO)",
                //   id: "icicle/rust-bindings/multi-gpu",
                // },
              ],
            },
          ],
        },
        {
          type: "doc",
          label: "Migrate from ICICLE v2",
          id: "icicle/migrate_from_v2",
        },
        {
          type: "doc",
          label: "Benchmarks",
          id: "icicle/benchmarks",
        },
        {
          type: "doc",
          label: "FAQ and Troubleshooting",
          id: "icicle/faq_and_troubleshooting",
        },
        {
          type: "doc",
          label: "Google Colab Instructions",
          id: "icicle/colab-instructions",
        },
        {
          type: "doc",
          label: "ICICLE Provers",
          id: "icicle/integrations"
        },
      ]
    },
    {
      type: "doc",
      label: "Ingonyama Grant program",
      id: "grants",
    },
    {
      type: "doc",
      label: "Contributor guide",
      id: "contributor-guide",
    },
    {
      type: "category",
      label: "Additional Resources",
      collapsed: false,
      collapsible: false,
      items: [
        {
          type: "link",
          label: "YouTube",
          href: "https://www.youtube.com/@ingo_ZK"
        },
        {
          type: "link",
          label: "Ingonyama Blog",
          href: "https://www.ingonyama.com/blog"
        },
        {
          type: "link",
          label: "Ingopedia",
          href: "https://www.ingonyama.com/ingopedia"
        },
        {
          href: 'https://github.com/ingonyama-zk',
          type: "link",
          label: 'GitHub',
        }
      ]
    }
  ],
};
