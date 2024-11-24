import type { Config } from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import type { NavbarItem } from '@docusaurus/theme-common';

import { themes } from 'prism-react-renderer';
import math from 'remark-math';
import katex from 'rehype-katex';

const lightCodeTheme = themes.github;
const darkCodeTheme = themes.dracula;

const ingoPreset = {
  docs: {
    showLastUpdateAuthor: true,
    showLastUpdateTime: true,
    includeCurrentVersion: false,
    routeBasePath: '/',
    remarkPlugins: [math],
    rehypePlugins: [katex],
    sidebarPath: require.resolve('./sidebars.ts'),
    editUrl: 'https://github.com/ingonyama-zk/icicle/tree/main/docs',
  },
  blog: {
    remarkPlugins: [math],
    rehypePlugins: [katex],
    showReadingTime: true,
    editUrl: 'https://github.com/ingonyama-zk/icicle/tree/main',
  },
  pages: {},
  theme: {
    customCss: require.resolve('./src/css/custom.css'),
  },
} satisfies Preset.Options

const navBarLeftSide = [
  {
    position: 'left',
    label: 'Docs',
    to: '/',
  }
] satisfies NavbarItem[]

const navBarRightSide = [
  {
    type: 'docsVersionDropdown',
    position: 'right',
    dropdownActiveClassDisabled: true,
  },
  {
    href: 'https://github.com/ingonyama-zk',
    position: 'right',
    label: 'GitHub',
  },
  {
    href: 'https://www.ingonyama.com/ingopedia/glossary',
    position: 'right',
    label: 'Ingopedia',
  },
  {
    type: 'dropdown',
    position: 'right',
    label: 'Community',
    items: [
      {
        label: 'Discord',
        href: 'https://discord.gg/6vYrE7waPj',
      },
      {
        label: 'Twitter',
        href: 'https://x.com/Ingo_zk',
      },
      {
        label: 'YouTube',
        href: 'https://www.youtube.com/@ingo_ZK'
      },
      {
        label: 'Mailing List',
        href: 'https://wkf.ms/3LKCbdj',
      }
    ]
  }
] satisfies NavbarItem[]

const config: Config = {
  title: 'Ingonyama Developer Documentation',
  tagline: 'Ingonyama is a next-generation semiconductor company, focusing on Zero-Knowledge Proof hardware acceleration. We build accelerators for advanced cryptography, unlocking real-time applications.',
  url: 'https://dev.ingonyama.com/',
  baseUrl: '/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/logo.png',
  organizationName: 'ingonyama-zk',
  projectName: 'developer-docs',
  trailingSlash: false,
  deploymentBranch: "main",
  staticDirectories: ['static'],
  presets: [
    [
      'classic',
      ingoPreset,
    ],
  ],
  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
      type: 'text/css',
      integrity:
        'sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM',
      crossorigin: 'anonymous',
    },
  ],

  scripts: [
    {
      src: 'https://plausible.io/js/script.js',
      'data-domain':'ingonyama.com',
      defer: true,
    },
  ],

  themeConfig:{
    metadata: [
      {name: 'twitter:card', content: 'summary_large_image'},
      {name: 'twitter:site', content: '@Ingo_zk'},
      {name: 'twitter:title', content: 'Ingonyama Developer Documentation'},
      {name: 'twitter:description', content: 'Ingonyama is a next-generation semiconductor company focusing on Zero-Knowledge Proof hardware acceleration...'},
      {name: 'twitter:image', content: 'https://dev.ingonyama.com/img/logo.png'},
      // title
      {name: 'og:title', content: 'Ingonyama Developer Documentation'},
      {name: 'og:description', content: 'Ingonyama is a next-generation semiconductor company focusing on Zero-Knowledge Proof hardware acceleration...'},
      {name: 'og:image', content: 'https://dev.ingonyama.com/img/logo.png'},
    ],
    docs: {
      sidebar: {
        hideable: true,
      }
    },
    colorMode: {
      defaultMode: 'dark',
      respectPrefersColorScheme: false,
    },
    algolia: {
      appId: 'PZY4KJBBBK',
      apiKey: '2cc940a6e0ef5c117f4f44e7f4e6e20b',
      indexName: 'ingonyama',
      contextualSearch: true,
      externalUrlRegex: 'external\\.com|domain\\.com',
      replaceSearchResultPathname: {
        from: '/docs/',
        to: '/',
      },
      searchParameters: {},
      searchPagePath: 'search',
    },
    navbar: {
      title: 'Ingonyama Developer Documentation',
      logo: {
        alt: 'Ingonyama Logo',
        src: 'img/logo.png',
      },
      items: [
        ...navBarLeftSide,
        ...navBarRightSide,
      ],
    },
    footer: {
      copyright: `Copyright © ${new Date().getFullYear()} Ingonyama, Inc. Built with Docusaurus.`,
    },
    prism: {
      theme: lightCodeTheme,
      darkTheme: darkCodeTheme,
      additionalLanguages: ['rust', 'go'],
    },
    image: 'img/logo.png',
    announcementBar: {
      id: 'announcement', // Any value that will identify this message.
      content:
        '<strong>❄️🎉 New Release! ICICLE v3.1! <a style="color:#000000;" target="_blank" rel="noopener noreferrer" href="https://medium.com/@ingonyama/icicle-v3-1-more-passion-more-energy-more-zk-performance-95c3aff4b295">Click here for the full update</a> 🎉❄️</strong>',
      backgroundColor: '#64f5ef', // Light blue background color.
      textColor: '#000000', // Black text color.
      isCloseable: true, // Defaults to `true`.
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
