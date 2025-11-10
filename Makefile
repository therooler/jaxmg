.PHONY: docs mkdocs-build deploy-docs

# Build the MkDocs site
mkdocs-build:
	mkdocs build

# Backwards-compatible alias
docs: mkdocs-build

# Build then deploy the built site to the gh-pages worktree
deploy-docs: mkdocs-build
	./tools/deploy_docs.sh
