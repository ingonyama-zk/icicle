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
    showLastUpdateAuthor: false,
    showLastUpdateTime: true,
    includeCurrentVersion: process.env.NODE_ENV !== 'production',
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
    activeBaseRegex: '^(?!\/api).*', // everything not starting with /api
  },
  {
    label: 'API',           
    to: '/apioverview',    
    position: 'left',
    activeBaseRegex: '^/api', // everything under the /api route
  },
  {
    type: 'html',
    position: 'left',
    value: `
      <a href="https://github.com/ingonyama-zk/icicle"
         class="github-cta-button"
         target="_blank"
         rel="noopener noreferrer">
        Go to ICICLE
      </a>
    `,
  },
] satisfies NavbarItem[];

const navBarRightSide = [

  {
    type: 'docsVersionDropdown',
    position: 'right',
    dropdownActiveClassDisabled: true,
  },

  {
    type: 'dropdown',
    position: 'right',
    label: 'Community',
    items: [
      { label: 'Discord', href: 'https://discord.gg/6vYrE7waPj' },
      { label: 'LinkedIn', href: 'https://www.linkedin.com/company/ingonyama' },
      { label: 'X/Twitter', href: 'https://x.com/Ingo_zk' },
      { label: 'YouTube', href: 'https://www.youtube.com/@ingo_ZK' },
      { label: 'Mailing List', href: 'https://wkf.ms/3LKCbdj' },
    ],
  },

  {
          label: 'Leave Feedback',
          position: 'right',
          href: 'https://forms.monday.com/forms/7b51e0bdad766b71da8869704c301472?r=use1',
          target: '_blank',
        },

  {
    type: 'html',
    position: 'right',
    value: `
      <a href="https://ingonyama.com"
        class="navbar__item ingo-paw"
        title="Visit Ingonyama"
        target="_blank"
        rel="noopener noreferrer">
        <img src="/img/Ingologo.svg" alt="Ingonyama logo" class="ingo-paw-logo" />
      </a>
    `,
  }

  
] satisfies NavbarItem[];


const config: Config = {
  title: 'ICICLE Docs',
  tagline: 'Explore the High-Speed Cryptography Library.',
  url: 'https://dev.ingonyama.com/',
  baseUrl: '/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: '/img/iciclelogo.png',
  organizationName: 'ingonyama-zk',
  projectName: 'developer-docs',
  trailingSlash: false,
  deploymentBranch: "main",
  staticDirectories: ['static'],
  presets: [
    [
      'classic',
      {
      ...ingoPreset,
        gtag: {
          trackingID: 'G-3XJCQFYEF9',
          anonymizeIP: true,
        },
      },
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
      {name: 'twitter:image', content: 'https://dev.ingonyama.com/img/ICICLELOGONEW.png'},
      // title
      {name: 'og:title', content: 'Ingonyama Developer Documentation'},
      {name: 'og:description', content: 'Ingonyama is a next-generation semiconductor company focusing on Zero-Knowledge Proof hardware acceleration...'},
      {name: 'og:image', content: 'https://dev.ingonyama.com/img/ICICLELOGONEW.png'},
    ],
      announcementBar: {
      id: 'my-special-announcement', // unique ID, change if you update the message
      content: '🚀 We just released <strong>ICICLE v4</strong> — featuring a more intuitive API with lattices. <a href="/start/integration-&-support/migrate_from_v3">Check out the migration guide</a>.',
      backgroundColor: '#006AEA', 
      textColor: '#ffffff',      
      isCloseable: true,          // allows users to dismiss the banner
    },
    docs: {
      sidebar: {
        hideable: true,
      }
    },
    colorMode: {
      defaultMode: 'light',
      disableSwitch: false,
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
      title: '',
      logo: {
        alt: 'Ingonyama Logo',
        src: '/img/icicledocslogo.png',
        className: 'custom-navbar-logo',
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
  } satisfies Preset.ThemeConfig,
};



export default config;
