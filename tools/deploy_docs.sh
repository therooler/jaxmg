#!/usr/bin/env bash
set -euo pipefail

# deploy_docs.sh - Build mkdocs site and push the `site` worktree to gh-pages
# Usage: ./tools/deploy_docs.sh [remote] [branch] [commit-message] [site-dir]
# Defaults: remote=origin, branch=gh-pages, commit-message="update site", site-dir=site

REMOTE=${1:-origin}
BRANCH=${2:-gh-pages}
COMMIT_MSG=${3:-"update site"}
SITE_DIR=${4:-site}

echo "[deploy_docs] remote=$REMOTE branch=$BRANCH site_dir=$SITE_DIR"

if ! command -v mkdocs >/dev/null 2>&1; then
  echo "mkdocs not found on PATH. Install mkdocs (pip install mkdocs) and try again." >&2
  exit 1
fi

echo "[deploy_docs] building site..."
mkdocs build

if [ ! -d "$SITE_DIR" ]; then
  echo "Site directory '$SITE_DIR' not found after build." >&2
  exit 1
fi

if ! git -C "$SITE_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  cat >&2 <<EOF
Directory '$SITE_DIR' is not a git worktree for this repository.
You can create the worktree with a command like:

  git worktree add -B $BRANCH $SITE_DIR $REMOTE/$BRANCH

This will create (or reset) the worktree branch to track the remote branch.
EOF
  exit 2
fi

echo "[deploy_docs] adding and committing changes in '$SITE_DIR'..."
cd "$SITE_DIR"

# Stage all changes (additions, deletions)
# Ensure GitHub Pages/Jekyll won't ignore underscore-prefixed folders (e.g. _static)
# Create an empty .nojekyll file in the site root — GitHub Pages will respect it and skip Jekyll processing.
touch .nojekyll

# Stage all changes (additions, deletions)
git add --all

if git diff --cached --quiet; then
  echo "[deploy_docs] no changes to commit in '$SITE_DIR'. Nothing to push."
  exit 0
fi

git commit -m "$COMMIT_MSG"

echo "[deploy_docs] pushing to $REMOTE/$BRANCH..."
# Push current HEAD to the requested branch on the remote
git push "$REMOTE" "HEAD:$BRANCH"

echo "[deploy_docs] done."
