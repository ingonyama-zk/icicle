# Contributor's Guide

We welcome all contributions with open arms. At Ingonyama we take a village approach, believing it takes many hands and minds to build an ecosystem.

## Contributing to ICICLE

- Make suggestions or report bugs via [GitHub issues](https://github.com/ingonyama-zk/icicle/issues)
- Contribute to the ICICLE by opening a [pull request](https://github.com/ingonyama-zk/icicle/pulls).
- Contribute to our [documentation](https://github.com/ingonyama-zk/icicle/tree/main/docs) and [examples](https://github.com/ingonyama-zk/icicle/tree/main/examples).
- Ask questions on [Discord](https://discord.gg/6vYrE7waPj)

### Opening a pull request

When opening a [pull request](https://github.com/ingonyama-zk/icicle/pulls) please keep the following in mind.

- **Clear Purpose** - The pull request should solve a single issue and be clean of any unrelated changes.
- **Clear Description** - If the pull request is for a new feature, describe what you built, why you added it, and how it's best that we test it. For bug fixes, please describe the issue and the solution.
- **Consistent Style** - Rust and Golang code should be linted by the official linters (`gofmt` and `rustfmt`) and maintain proper style. For CUDA and C++ code, we use [`clang-format`](https://github.com/ingonyama-zk/icicle/blob/main/.clang-format). See [here](https://github.com/ingonyama-zk/icicle/blob/605c25f9d22135c54ac49683b710fe2ce06e2300/.github/workflows/main-format.yml#L46) for how we run it.
- **Minimal Tests** - Please add tests that cover basic usage of your changes.

## Contributor License Agreement (CLA)

Before we can merge your contribution, you must agree to the terms of our Contributor License Agreement:

### 1. Grant of Copyright License

You grant Ingonyama and its licensees a perpetual, worldwide, non-exclusive, royalty-free, irrevocable license under your copyright rights in your contribution:
- To use, reproduce, prepare derivative works of, publicly display, publicly perform, sublicense, and distribute your contribution and derivative works.

### 2. Grant of Patent License

You grant Ingonyama and its licensees a perpetual, worldwide, non-exclusive, royalty-free, irrevocable (except as stated below) license under patent claims that you control to make, use, sell, offer for sale, import, and otherwise transfer your contribution.

This license does not extend to claims infringed by modifications or derivative works of your contribution not made or authorized by you.

### 3. Your Contribution

You confirm that:
- You are legally entitled to submit your contribution.
- Your contribution is your original creation and does not violate any third-party rights.
- If applicable, you have secured your employer’s permission or your employer has signed a separate agreement with us.

### 4. No Obligation to Use

Submitting a contribution does not guarantee inclusion in the ICICLE codebase.

### 5. Identification

Your contribution may be associated with your name and/or GitHub handle unless you request anonymity.

### 6. Entire Agreement

This agreement is the complete agreement between you and Ingonyama regarding your contribution.

> If you are contributing on behalf of an organization, please contact us at [hi@ingonyama.com](mailto:hi@ingonyama.com) to arrange a Corporate CLA.

## Questions?

Find us on [Discord](https://discord.gg/6vYrE7waPj).
