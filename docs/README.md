# ICICLE Developer Docs

The developer docs for ICICLE is a static website built using [Docusaurus](https://docusaurus.io/).

## Requirements

Docusaurus is written in Typescript and distributed as npm packages. npm is a prerequisite as is node.js

If node.js or npm aren't installed, its suggested to use [nvm](https://github.com/nvm-sh/nvm?tab=readme-ov-file#installing-and-updating) to [install both](https://github.com/nvm-sh/nvm?tab=readme-ov-file#usage) at the same time.

## Install

```
npm install
```

## Versioning

ICICLE docs are versioned, keeping the latest set of docs for previous major versions and the latest 4 sets of docs for the current major version.

The [docs](./docs/) directory holds the next version's docs
All **released** versions are under the [versioned_docs](./versioned_docs/) directory.

### Releasing new versions

In order to create a new version, run the following:

```sh
npm run docusaurus docs:version <version to create>
```

For example:

Assuming the next version is 5.6.0, we would run the following:

```sh
npm run docusaurus docs:version 5.6.0
```

This command will:

1. Add a new version for the specified `<version to create>` in the [versions file](./versions.json)
2. Create a directory under [versioned_docs](./versioned_docs/) with the name `version-<version to create>` and copies everything in the [docs](./docs/) directory there.
3. Create a file under [versioned_sidebars](./versioned_sidebars/) with the name `version-<version to create>-sidebars.json` and copies the current [sidebar.ts](./sidebars.ts) file there after converting it to a json object.

### Removing old versions

1. Remove the version from versions.json.

[
  "3.2.0",
  "3.1.0",
  "3.0.0",
  "2.8.0",
- "1.10.1"
]


2. Delete the versioned docs directory for that version. Example: versioned_docs/version-1.10.1.
3. Delete the versioned sidebars file. Example: versioned_sidebars/version-1.10.1-sidebars.json.

## Local development

To render the site, run the following

```
npm start
```

This command starts a local development server and opens up a browser window on port 3000. Most changes are reflected live (hot reloaded) without having to restart the server.

By default, the next version's docs are not rendered. In order to view any changes in the next version's docs, update the following in the [config file](./docusaurus.config.ts):

```ts
const ingoPreset = {
  docs: {
    .
    .
    includeCurrentVersion: false, // Update this to true to render the next verion's docs
    .
    .
    .

  },
  .
  .
  .
} satisfies Preset.Options
```

### Updating docs in old versions

In order to update docs for old versions, the files under the specific version's [versioned_docs](./versioned_docs/) directory must be updated.

### Updating docs across versions

If docs need updating across multiple versions, including future versions, they need to be updated in each previous version's [versioned_docs](./versioned_docs/) and the next version's docs under the [docs](./docs/) directory
