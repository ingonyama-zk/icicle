# Contributor's Guide

We welcome all contributions with open arms. At Ingonyama we take a village approach, believing it takes many hands and minds to build an ecosystem.

## Contributing to ICICLE

- Make suggestions or report bugs via [GitHub issues](https://github.com/ingonyama-zk/icicle/issues)
- Contribute to the ICICLE by opening a [pull request](https://github.com/ingonyama-zk/icicle/pulls).
- Contribute to our [documentation](https://github.com/ingonyama-zk/icicle/tree/main/docs) and [examples](https://github.com/ingonyama-zk/icicle/tree/main/examples).
- Ask questions on Discord

### Opening a pull request

When opening a [pull request](https://github.com/ingonyama-zk/icicle/pulls) please keep the following in mind.

- `Clear Purpose` - The pull request should solve a single issue and be clean of any unrelated changes.
- `Clear description` - If the pull request is for a new feature describe what you built, why you added it and how it's best that we test it. For bug fixes please describe the issue and the solution.
- `Consistent style` - Rust and Golang code should be linted by the official linters (golang fmt and rust fmt) and maintain a proper style. For CUDA and C++ code we use [`clang-format`](https://github.com/ingonyama-zk/icicle/blob/main/.clang-format), [here](https://github.com/ingonyama-zk/icicle/blob/605c25f9d22135c54ac49683b710fe2ce06e2300/.github/workflows/main-format.yml#L46) you can see how we run it.
- `Minimal Tests` - please add test which cover basic usage of your changes.

## Questions?

Find us on [Discord](https://discord.gg/6vYrE7waPj).
