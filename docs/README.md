# ICICLE Developer Docs

The developer docs for ICICLE is a static website built using [Docusaurus](https://docusaurus.io/).

## Requirements

Docusaurus is written in Typescript and distributed as npm packages. npm is a prerequisite as is node.js

If node.js or npm aren't installed, its suggested to use [nvm](https://github.com/nvm-sh/nvm?tab=readme-ov-file#installing-and-updating) to [install both](https://github.com/nvm-sh/nvm?tab=readme-ov-file#usage) at the same time.

## Install

```sh
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

- Remove the version from versions.json.

```json
  [
    "3.2.0",
    "3.1.0",
    "3.0.0",
    "2.8.0",
  - "1.10.1"
  ]
```

- Delete the versioned docs directory for that version. Example: versioned_docs/version-1.10.1.
- Delete the versioned sidebars file. Example: versioned_sidebars/version-1.10.1-sidebars.json.

## Static assets

Static assets like images should be placed in the top level [static](./static/) directory **regardless** of which version it will be used in.

Docusaurus adds all of the files in the directories listed as `staticDirectories` in the config to the root of the build output so they can be accessed directly from the root path.

Read more on this [here](https://docusaurus.io/docs/static-assets)

### Adding a new static directory

To update where Docusaurus looks for static directories, add the directory name to the `statidDirectories` list in the config:

```ts
const config: Config = {
  .
  .
  .
  staticDirectories: ['static'/*, "another static dir" */],
  .
  .
  .
}
```

### Linking to static assets in docs

Since the static assets are at the root of the build output, static assets can be linked to directly from the root, maintaining the directory hierarchy they have in the static directory.

For example:

If an image is located at `static/images/poseidon.png`, it should be linked to as `/images/poseidon.png`

## Local development

To render the site, run the following

```sh
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

## Adding or removing docs from rendering

Each version has its own sidebar.json file located in the [versioned_sidebars](./versioned_sidebars/) directory.

The next version's sidebar is found in [sidebar.ts](./sidebars.ts).

You can add or remove a doc from there to change the sidebar and include or prevent docs from rendering.

## Troubleshooting

### Latex isn't rendering correctly

Latex formula must have the `$$` on a separate line:

```mdx
$$
M_{4} = \begin{pmatrix}
5 & 7 & 1 & 3 \\
4& 6 & 1 & 1 \\
1 & 3 & 5 & 7\\
1 & 1 & 4 & 6\\
\end{pmatrix}
$$
```
