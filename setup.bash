#!/bin/bash

echo "📦 Installing pre-commit..."
pip install pre-commit

echo "⚙️ Installing pre-commit hooks..."
pre-commit install --install-hooks

echo "🔐 Setting up Git pre-push hook..."
HOOK_PATH=".git/hooks/pre-push"
mkdir -p "$(dirname "$HOOK_PATH")"
cp .githooks/pre-push "$HOOK_PATH"
chmod +x "$HOOK_PATH"

echo "✅ Setup complete! Pre-commit will now run on commit and push."
