// @ts-check
const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');
const math = require('remark-math');
const katex = require('rehype-katex');

/** @type {import('@docusaurus/types').Config} */
const config = {
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
  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          showLastUpdateAuthor: true,
          showLastUpdateTime: true,
          routeBasePath: '/',
          remarkPlugins: [math, require('mdx-mermaid')],
          rehypePlugins: [katex],
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/ingonyama-zk/icicle/tree/main/docs',
        },
        blog: {
          remarkPlugins: [math, require('mdx-mermaid')],
          rehypePlugins: [katex],
          showReadingTime: true,
          editUrl: 'https://github.com/ingonyama-zk/icicle/tree/main',
        },
        pages: {},
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
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

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
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
      hideableSidebar: true,
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
          {
            position: 'left',
            label: 'Docs',
            to: '/',
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
          },

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
    }),
};

module.exports = config;

