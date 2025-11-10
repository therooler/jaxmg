# Building from source
## Build from source

To build from source:

```bash
mkdir build
cd build
cmake ..
cmake --build . --target install
```

This installs the CUDA binaries into src/jaxmg/bin.

Dependencies are managed with [CPM-CMAKE](https://github.com/cpm-cmake/CPM.cmake),
including **abseil-cpp**, **jaxlib**, **XLA** for compilation. Compilation requires C++20 or later.
To build specific targets only, for example potrs:
```bash
cmake ..
cmake --build . --target potrs && cmake --install .
```

## Docs setup

https://olgarithms.github.io/sphinx-tutorial/docs/7-hosting-on-github-pages.html

# Deploying the MkDocs site

This repository includes a small helper script and Makefile target to build and publish the MkDocs-generated `site/` directory to the `gh-pages` branch.

Quick steps (one-time):

1. Ensure the `gh-pages` worktree exists locally:

   git worktree add -B gh-pages site origin/gh-pages

2. Build and push the site:

   make deploy-docs

Or run the script directly:

   ./tools/deploy_docs.sh origin gh-pages "update site"

What the script does:

- Runs `mkdocs build` (so make sure mkdocs is installed)
- Verifies that `site/` is a git worktree
- Stages all changes in `site/`, commits with the message provided, and pushes to `remote/branch` (defaults: `origin/gh-pages`)

Notes and tips:

- If `site` is not yet a worktree the script will print a suggestion and exit. Create the worktree with the one-time `git worktree add` command above.
- The script intentionally refuses to create the worktree automatically to avoid surprising changes to your git state.
- For CI-based publishing, consider a GitHub Actions workflow that installs Python, dependencies, runs `mkdocs build`, and pushes the `site` tree to `gh-pages`.
