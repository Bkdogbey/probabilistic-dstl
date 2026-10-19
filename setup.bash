#!/bin/bash

echo "📦 Installing pre-commit..."
pip install pre-commit

echo "⚙️ Installing pre-commit hooks..."
pre-commit install --install-hooks

echo "✅ Setup complete! Pre-commit will now run on commit."
