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
          type: "doc",
          label: "Getting started",
          id: "icicle/introduction"
        },
        {
          type: "doc",
          label: "ICICLE Core",
          id: "icicle/core",
        },
        {
          type: "category",
          label: "Primitives",
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
              label: "NTT",
              id: "icicle/primitives/ntt",
            },
            {
              type: "doc",
              label: "Keccak Hash",
              id: "icicle/primitives/keccak",
            },
            {
              type: "doc",
              label: "Poseidon Hash",
              id: "icicle/primitives/poseidon",
            },
          ],
        },
        {
          type: "doc",
          label: "Polynomials",
          id: "icicle/polynomials/overview",
        },
        {
          type: "doc",
          label: "Multi GPU Support",
          id: "icicle/multi-gpu",
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
              label: "Keccak Hash",
              id: "icicle/golang-bindings/keccak",
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
              type: "category",
              label: "MSM",
              link: {
                type: `doc`,
                id: "icicle/rust-bindings/msm",
              },
              collapsed: true,
              items: [
                {
                  type: "doc",
                  label: "MSM pre computation",
                  id: "icicle/rust-bindings/msm-pre-computation",
                }
              ]
            },
            {
              type: "doc",
              label: "NTT",
              id: "icicle/rust-bindings/ntt",
            },
            {
              type: "doc",
              label: "EC-NTT",
              id: "icicle/rust-bindings/ecntt",
            },
            {
              type: "doc",
              label: "Vector operations",
              id: "icicle/rust-bindings/vec-ops",
            },
            {
              type: "doc",
              label: "Keccak Hash",
              id: "icicle/rust-bindings/keccak",
            },
            {
              type: "doc",
              label: "Multi GPU Support",
              id: "icicle/rust-bindings/multi-gpu",
            },
            {
              type: "doc",
              label: "Polynomials",
              id: "icicle/rust-bindings/polynomials",
            },
          ],
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
      label: "ZK Containers",
      id: "ZKContainers",
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
